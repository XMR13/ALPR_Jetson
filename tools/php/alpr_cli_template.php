<?php
/**
 * ALPR CLI bridge template for PHP.
 *
 * The helper now keeps a long-lived `e2e-json-stream` process warm so repeated
 * requests avoid model reloads. Behaviour is backwards compatible: callers still
 * invoke `run_alpr()` per image and receive the same response structure.
 *
 * If streaming is undesirable (e.g., when requesting annotated outputs), set
 * `USE_STREAM=0` in `$envOverrides` or export `ALPR_PHP_USE_STREAM=0` and the
 * code will fall back to the legacy one-shot wrapper.
 */

declare(strict_types=1);

const REPO_ROOT = __DIR__ . '/..';
const DEFAULT_DET_ENGINE = REPO_ROOT . '/models/detector/yolov9-s_plate_fp16.engine';
const DEFAULT_OCR_ONNX = REPO_ROOT . '/models/ocr/cct_s_v1_global.onnx';
const DEFAULT_PLATE_CONFIG = REPO_ROOT . '/models/ocr/cct_s_v1_global_plate_config.yaml';
const DEFAULT_OCR_ENGINE = REPO_ROOT . '/models/ocr/ppo_crnn_fp16.engine';
const DEFAULT_CHARSET = REPO_ROOT . '/models/ocr/charset.txt';

/**
 * Execute the ALPR pipeline for a single image.
 *
 * @param string $imagePath Absolute path to the image file.
 * @param bool   $textOnly  When true, return text-only output (exit code 3 => no/invalid plate).
 * @param array<string,mixed> $envOverrides Extra environment-style overrides (model paths, feature flags).
 *
 * @return array{ok:bool, exit_code:int, stdout:string, stderr?:string, data?:array<string,mixed>}
 */
function run_alpr(string $imagePath, bool $textOnly = false, array $envOverrides = []): array
{
    if (!is_file($imagePath)) {
        return [
            'ok' => false,
            'exit_code' => 2,
            'stdout' => '',
            'stderr' => 'image not found: ' . $imagePath,
        ];
    }

    $env = array_merge(default_env(), $envOverrides);
    $effectiveTextOnly = $textOnly || normalize_bool($envOverrides['TEXT_ONLY'] ?? ($env['TEXT_ONLY'] ?? '0'));
    if ($effectiveTextOnly) {
        $env['TEXT_ONLY'] = '0'; // keep JSON output; we handle text conversion ourselves.
    } else {
        unset($env['TEXT_ONLY']);
    }

    if (should_use_stream($env, $effectiveTextOnly)) {
        try {
            return run_alpr_stream($imagePath, $effectiveTextOnly, $env);
        } catch (RuntimeException $streamError) {
            $fallback = run_alpr_once($imagePath, $effectiveTextOnly, $env);
            $fallback['stderr'] = trim((string)($fallback['stderr'] ?? '') . "\nstream_fallback: " . $streamError->getMessage());
            return $fallback;
        }
    }

    return run_alpr_once($imagePath, $effectiveTextOnly, $env);
}

/**
 * Return the default environment overrides applied before user input.
 *
 * @return array<string,string>
 */
function default_env(): array
{
    $pythonBin = getenv('PYTHON_BIN');
    return [
        'PYTHONPATH' => REPO_ROOT . '/src',
        'PYTHONUNBUFFERED' => '1',
        'PYTHON_BIN' => $pythonBin !== false && $pythonBin !== '' ? $pythonBin : 'python3',
        'DET_ENGINE' => DEFAULT_DET_ENGINE,
        'OCR_BACKEND' => 'onnx',
        'OCR_ONNX' => DEFAULT_OCR_ONNX,
        'PLATE_CONFIG' => DEFAULT_PLATE_CONFIG,
        'OCR_ENGINE' => DEFAULT_OCR_ENGINE,
        'CHARSET' => DEFAULT_CHARSET,
        'CONF' => '0.5',
        'IOU' => '0.45',
        'MIN_PLATE_H' => '28',
        'MIN_AR' => '1.5',
        'MAX_AR' => '5.0',
        'TOPK' => '1',
        'POSTPROC' => 'indonesia',
        'ACCEPT_ALL' => '1',
    ];
}

/**
 * Determine whether to use the persistent stream process.
 *
 * @param array<string,mixed> $env
 */
function should_use_stream(array $env, bool $textOnly): bool
{
    if (!empty($env['ANNOTATE_DIR'])) {
        return false; // wrapper handles annotation sidecar calls.
    }
    if (array_key_exists('USE_STREAM', $env)) {
        return normalize_bool($env['USE_STREAM'], true);
    }
    $global = getenv('ALPR_PHP_USE_STREAM');
    if ($global !== false && $global !== '') {
        return normalize_bool($global, true);
    }
    return true;
}

/**
 * Persistent NDJSON stream runner.
 *
 * @param array<string,mixed> $env
 * @return array{ok:bool, exit_code:int, stdout:string, stderr?:string, data?:array<string,mixed>}
 */
function run_alpr_stream(string $imagePath, bool $textOnly, array $env): array
{
    $config = resolve_stream_command($env);
    $client = AlprStreamClient::instance($config['command'], $config['fingerprint']);
    [$jsonLine, $stderr] = $client->process($imagePath);

    $payload = json_decode($jsonLine, true);
    if (!is_array($payload)) {
        throw new RuntimeException('failed to decode JSON output');
    }

    if ($textOnly) {
        [$exitCode, $textOut] = resolve_text_only($payload, $env);
        handle_text_files($env, $textOut, $exitCode);
        return [
            'ok' => $exitCode === 0,
            'exit_code' => $exitCode,
            'stdout' => $textOut,
            'stderr' => $stderr,
            'data' => $payload,
        ];
    }

    return [
        'ok' => true,
        'exit_code' => 0,
        'stdout' => $jsonLine,
        'stderr' => $stderr,
        'data' => $payload,
    ];
}

/**
 * Legacy one-shot wrapper fallback.
 *
 * @param array<string,mixed> $env
 * @return array{ok:bool, exit_code:int, stdout:string, stderr?:string, data?:array<string,mixed>}
 */
function run_alpr_once(string $imagePath, bool $textOnly, array $env): array
{
    $wrapper = REPO_ROOT . '/tools/alpr_e2e_json.sh';
    if (!is_file($wrapper) || !is_executable($wrapper)) {
        return [
            'ok' => false,
            'exit_code' => 2,
            'stdout' => '',
            'stderr' => 'wrapper not found or not executable: ' . $wrapper,
        ];
    }

    $envAssign = build_env_assignments($env, ['USE_STREAM', 'TEXT_ONLY']);
    $commandParts = [];
    if (!empty($envAssign)) {
        $commandParts[] = implode(' ', $envAssign);
    }
    $commandParts[] = escapeshellcmd($wrapper);
    $commandParts[] = escapeshellarg($imagePath);
    $cmd = implode(' ', $commandParts);

    $descriptorSpec = [
        1 => ['pipe', 'w'],
        2 => ['pipe', 'w'],
    ];

    $process = proc_open($cmd, $descriptorSpec, $pipes, REPO_ROOT);
    if (!is_resource($process)) {
        return [
            'ok' => false,
            'exit_code' => 2,
            'stdout' => '',
            'stderr' => 'failed to launch process',
        ];
    }

    $stdout = stream_get_contents($pipes[1]);
    $stderr = stream_get_contents($pipes[2]);
    fclose($pipes[1]);
    fclose($pipes[2]);
    $exitCode = proc_close($process);

    if ($exitCode !== 0) {
        return [
            'ok' => false,
            'exit_code' => $exitCode,
            'stdout' => (string)$stdout,
            'stderr' => $stderr,
        ];
    }

    $payload = json_decode((string)$stdout, true);
    if (!is_array($payload)) {
        return [
            'ok' => false,
            'exit_code' => 2,
            'stdout' => (string)$stdout,
            'stderr' => $stderr !== '' ? $stderr : 'failed to decode JSON output',
        ];
    }

    if ($textOnly) {
        [$rc, $textOut] = resolve_text_only($payload, $env);
        handle_text_files($env, $textOut, $rc);
        return [
            'ok' => $rc === 0,
            'exit_code' => $rc,
            'stdout' => $textOut,
            'stderr' => $stderr,
            'data' => $payload,
        ];
    }

    return [
        'ok' => true,
        'exit_code' => 0,
        'stdout' => trim((string)$stdout),
        'stderr' => $stderr,
        'data' => $payload,
    ];
}

/**
 * Convert payload to text-only response, mirroring the shell wrapper semantics.
 *
 * @param array<string,mixed> $payload
 * @return array{int,string} [exitCode, text]
 */
function resolve_text_only(array $payload, array $env): array
{
    $placeholder = isset($env['TEXT_NO_PLATE']) ? (string)$env['TEXT_NO_PLATE'] : '';
    if (($payload['status'] ?? '') !== 'ok') {
        return [3, $placeholder];
    }

    $plates = $payload['plates'] ?? [];
    if (!is_array($plates) || empty($plates)) {
        return [3, $placeholder];
    }

    $best = $plates[0];
    $mode = strtolower((string)($env['TEXT_MODE'] ?? 'best'));
    $allowInvalid = normalize_bool($env['TEXT_ALLOW_INVALID'] ?? '0');
    $textBest = trim((string)($best['text'] ?? ''));
    $textRaw = trim((string)($best['ocr_raw'] ?? ''));
    $valid = !empty($best['valid']);

    if ($mode === 'raw') {
        if ($textRaw !== '') {
            return [0, $textRaw];
        }
        if ($allowInvalid && $textBest !== '') {
            return [0, $textBest];
        }
        return [3, $placeholder];
    }

    if ($textBest !== '' && ($valid || $allowInvalid)) {
        return [0, $textBest];
    }
    if ($allowInvalid && $textRaw !== '') {
        return [0, $textRaw];
    }

    return [3, $placeholder];
}

/**
 * Persist optional TEXT_OUT_FILE / TEXT_RC_FILE outputs for text-only mode.
 *
 * @param array<string,mixed> $env
 */
function handle_text_files(array $env, string $text, int $exitCode): void
{
    if (isset($env['TEXT_RC_FILE']) && $env['TEXT_RC_FILE'] !== '') {
        ensure_parent_dir((string)$env['TEXT_RC_FILE']);
        @file_put_contents((string)$env['TEXT_RC_FILE'], $exitCode . PHP_EOL);
    }

    if ($exitCode === 0 && isset($env['TEXT_OUT_FILE']) && $env['TEXT_OUT_FILE'] !== '') {
        ensure_parent_dir((string)$env['TEXT_OUT_FILE']);
        @file_put_contents((string)$env['TEXT_OUT_FILE'], $text . PHP_EOL);
    }
}

/**
 * Resolve the NDJSON stream command and fingerprint for caching.
 *
 * @param array<string,mixed> $env
 * @return array{command:string,fingerprint:string}
 */
function resolve_stream_command(array $env): array
{
    $parts = [
        escapeshellcmd((string)$env['PYTHON_BIN']),
        '-m',
        'alpr_jetson',
        'e2e-json-stream',
    ];

    $detEngine = (string)($env['DET_ENGINE'] ?? DEFAULT_DET_ENGINE);
    if (!is_file($detEngine)) {
        throw new RuntimeException('detector engine not found: ' . $detEngine);
    }
    $parts[] = '--det-engine';
    $parts[] = escapeshellarg($detEngine);

    $parts[] = '--conf';
    $parts[] = escapeshellarg((string)$env['CONF']);
    $parts[] = '--iou';
    $parts[] = escapeshellarg((string)$env['IOU']);
    $parts[] = '--min-plate-h';
    $parts[] = escapeshellarg((string)$env['MIN_PLATE_H']);
    $parts[] = '--min-ar';
    $parts[] = escapeshellarg((string)$env['MIN_AR']);
    $parts[] = '--max-ar';
    $parts[] = escapeshellarg((string)$env['MAX_AR']);
    $parts[] = '--topk';
    $parts[] = escapeshellarg((string)$env['TOPK']);

    $backend = strtolower((string)$env['OCR_BACKEND']);
    if ($backend === 'trt') {
        $ocrEngine = (string)($env['OCR_ENGINE'] ?? DEFAULT_OCR_ENGINE);
        $charset = (string)($env['CHARSET'] ?? DEFAULT_CHARSET);
        if (!is_file($ocrEngine)) {
            throw new RuntimeException('OCR TensorRT engine not found: ' . $ocrEngine);
        }
        if (!is_file($charset)) {
            throw new RuntimeException('charset not found: ' . $charset);
        }
        $parts[] = '--engine';
        $parts[] = escapeshellarg($ocrEngine);
        $parts[] = '--charset';
        $parts[] = escapeshellarg($charset);
    } else {
        $ocrOnnx = (string)($env['OCR_ONNX'] ?? DEFAULT_OCR_ONNX);
        $plateCfg = (string)($env['PLATE_CONFIG'] ?? DEFAULT_PLATE_CONFIG);
        if (!is_file($ocrOnnx)) {
            throw new RuntimeException('OCR ONNX model not found: ' . $ocrOnnx);
        }
        if (!is_file($plateCfg)) {
            throw new RuntimeException('plate_config not found: ' . $plateCfg);
        }
        $parts[] = '--onnx';
        $parts[] = escapeshellarg($ocrOnnx);
        $parts[] = '--plate-config';
        $parts[] = escapeshellarg($plateCfg);
        $parts[] = '--onnx-provider';
        $parts[] = 'cuda';
        $memLimit = (string)($env['ONNX_GPU_MEM_LIMIT_MB'] ?? $env['ONNX_GPU_MEM_LIMIT'] ?? '512');
        if ($memLimit !== '') {
            $parts[] = '--onnx-gpu-mem-limit-mb';
            $parts[] = escapeshellarg($memLimit);
        }
    }

    $parts[] = '--postproc';
    $parts[] = escapeshellarg((string)$env['POSTPROC']);

    $allowed = parse_allowed_prefix($env['ALLOWED_PREFIX'] ?? null);
    if (!empty($allowed)) {
        $parts[] = '--allowed-prefix';
        foreach ($allowed as $prefix) {
            $parts[] = escapeshellarg($prefix);
        }
    }

    if (normalize_bool($env['ACCEPT_ALL'] ?? '1', true)) {
        $parts[] = '--accept-all';
    }
    if (normalize_bool($env['DEBUG_CROPS'] ?? '0')) {
        $parts[] = '--debug-crops';
    }
    if (normalize_bool($env['STOP_ON_ERROR'] ?? '0')) {
        $parts[] = '--stop-on-error';
    }

    $envAssign = build_env_assignments($env, ['PYTHON_BIN', 'USE_STREAM', 'TEXT_ONLY']);
    $command = trim(implode(' ', array_filter([
        implode(' ', $envAssign),
        implode(' ', $parts),
    ])));

    return [
        'command' => $command,
        'fingerprint' => sha1($command),
    ];
}

/**
 * Normalise various truthy/falsey inputs.
 *
 * @param mixed $value
 */
function normalize_bool($value, bool $default = false): bool
{
    if (is_bool($value)) {
        return $value;
    }
    if (is_int($value)) {
        return $value !== 0;
    }
    if (is_string($value)) {
        $normalized = strtolower(trim($value));
        if ($normalized === '') {
            return $default;
        }
        return in_array($normalized, ['1', 'true', 'yes', 'on', 'y'], true);
    }
    return $default;
}

/**
 * Map env array to shell env assignments.
 *
 * @param array<string,mixed> $env
 * @return list<string>
 */
function build_env_assignments(array $env, array $exclude = []): array
{
    $skip = array_flip($exclude);
    $assignments = [];
    foreach ($env as $key => $value) {
        if (isset($skip[$key]) || $value === null) {
            continue;
        }
        $assignments[] = sprintf('%s=%s', $key, escapeshellarg((string)$value));
    }
    return $assignments;
}

/**
 * Parse allowed prefix env (comma/space separated or array).
 *
 * @param mixed $value
 * @return list<string>
 */
function parse_allowed_prefix($value): array
{
    if (is_array($value)) {
        $items = $value;
    } elseif (is_string($value)) {
        $value = trim($value);
        if ($value === '') {
            return [];
        }
        $items = preg_split('/[,\s]+/', $value) ?: [];
    } else {
        return [];
    }

    $out = [];
    foreach ($items as $item) {
        $candidate = strtoupper(trim((string)$item));
        if ($candidate !== '') {
            $out[] = $candidate;
        }
    }
    return array_values(array_unique($out));
}

/**
 * Ensure parent directory exists before writing helper files.
 */
function ensure_parent_dir(string $filePath): void
{
    $dir = dirname($filePath);
    if ($dir !== '' && !is_dir($dir)) {
        @mkdir($dir, 0775, true);
    }
}

/**
 * Lightweight wrapper around the long-lived stream process.
 */
final class AlprStreamClient
{
    /** @var ?self */
    private static $instance = null;
    private static string $fingerprint = '';

    /** @var resource|null */
    private $process = null;
    /** @var array<int,resource> */
    private array $pipes = [];
    private string $command;

    private function __construct(string $command)
    {
        $this->command = $command;
        $this->launch();
    }

    public function __destruct()
    {
        $this->close();
    }

    public static function instance(string $command, string $fingerprint): self
    {
        if (!self::$instance || self::$fingerprint !== $fingerprint) {
            if (self::$instance) {
                self::$instance->close();
            }
            self::$instance = new self($command);
            self::$fingerprint = $fingerprint;
            register_shutdown_function(static function (): void {
                if (self::$instance) {
                    self::$instance->close();
                }
            });
        }
        return self::$instance;
    }

    /**
     * @return array{string,string} [jsonLine, stderr]
     */
    public function process(string $imagePath): array
    {
        $attempts = 0;
        do {
            try {
                return $this->invoke($imagePath);
            } catch (RuntimeException $err) {
                $this->restart();
                $attempts++;
                if ($attempts >= 2) {
                    throw $err;
                }
            }
        } while (true);
    }

    private function invoke(string $imagePath): array
    {
        $this->ensureProcess();
        if (!is_resource($this->pipes[0])) {
            throw new RuntimeException('stream stdin unavailable');
        }
        if (@fwrite($this->pipes[0], $imagePath . PHP_EOL) === false) {
            throw new RuntimeException('failed to write path to stream');
        }
        fflush($this->pipes[0]);
        $line = $this->readLine();
        $stderr = $this->drainStderr();
        return [$line, $stderr];
    }

    private function readLine(): string
    {
        if (!is_resource($this->pipes[1])) {
            throw new RuntimeException('stream stdout unavailable');
        }
        while (true) {
            $line = stream_get_line($this->pipes[1], 1_048_576, "\n");
            if ($line === false) {
                if (feof($this->pipes[1])) {
                    throw new RuntimeException('stream stdout closed unexpectedly');
                }
                usleep(1_000);
                continue;
            }
            $trimmed = trim($line);
            if ($trimmed === '') {
                continue;
            }
            return $trimmed;
        }
    }

    private function drainStderr(): string
    {
        if (!is_resource($this->pipes[2])) {
            return '';
        }
        $data = stream_get_contents($this->pipes[2]);
        return $data === false ? '' : $data;
    }

    private function ensureProcess(): void
    {
        if (!is_resource($this->process)) {
            $this->launch();
            return;
        }
        $status = proc_get_status($this->process);
        if (!$status['running']) {
            $this->restart();
        }
    }

    private function launch(): void
    {
        $descriptorSpec = [
            0 => ['pipe', 'w'],
            1 => ['pipe', 'r'],
            2 => ['pipe', 'r'],
        ];
        $process = proc_open($this->command, $descriptorSpec, $pipes, REPO_ROOT);
        if (!is_resource($process)) {
            throw new RuntimeException('failed to launch stream process');
        }
        $this->process = $process;
        $this->pipes = $pipes;
        stream_set_blocking($this->pipes[0], true);
        stream_set_blocking($this->pipes[1], true);
        stream_set_blocking($this->pipes[2], false);
    }

    private function restart(): void
    {
        $this->close();
        $this->launch();
    }

    private function close(): void
    {
        foreach ($this->pipes as $pipe) {
            if (is_resource($pipe)) {
                fclose($pipe);
            }
        }
        $this->pipes = [];
        if (is_resource($this->process)) {
            proc_close($this->process);
        }
        $this->process = null;
    }
}

// Example: minimal handler for an uploaded file (POST form with input name "image").
if (php_sapi_name() !== 'cli' && ($_SERVER['REQUEST_METHOD'] ?? '') === 'POST') {
    if (!isset($_FILES['image']) || $_FILES['image']['error'] !== UPLOAD_ERR_OK) {
        http_response_code(400);
        echo json_encode(['error' => 'missing or invalid image upload']);
        exit;
    }

    $tmpPath = $_FILES['image']['tmp_name'];
    $result = run_alpr($tmpPath);
    header('Content-Type: application/json');
    if ($result['ok']) {
        echo json_encode($result['data']);
    } else {
        http_response_code(500);
        echo json_encode([
            'error' => 'alpr_failed',
            'detail' => $result,
        ]);
    }
    exit;
}
