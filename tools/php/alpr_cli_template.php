<?php
/**
 * ALPR CLI bridge template for PHP.
 *
 * This script demonstrates how to invoke the repository’s one-shot wrapper
 * (`tools/alpr_e2e_json.sh`) from PHP. It expects the Jetson-side models to be
 * mounted in their default locations (see README.md) and returns either the
 * parsed JSON payload or a structured error object.
 *
 * Usage (development):
 *
 *   $result = run_alpr(__DIR__ . '/../samples/frame.jpg');
 *   if ($result['ok']) {
 *       var_dump($result['data']);
 *   } else {
 *       error_log('ALPR failed: ' . json_encode($result));
 *   }
 *
 * Integrate this helper with your upload handler (e.g. $_FILES['image']) and
 * replace the default model paths if they differ on your system.
 */

declare(strict_types=1);

const REPO_ROOT = __DIR__ . '/..';
/**
 * Execute the ALPR CLI wrapper on a local image.
 *
 * @param string $imagePath Absolute path to the image file.
 * @param bool   $textOnly  When true, request text-only output (exit code 3 => no plate).
 * @param array<string,string> $envOverrides Additional environment variables for the subprocess.
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

    $wrapper = REPO_ROOT . '/tools/alpr_e2e_json.sh';
    if (!is_file($wrapper) || !is_executable($wrapper)) {
        return [
            'ok' => false,
            'exit_code' => 2,
            'stdout' => '',
            'stderr' => 'wrapper not found or not executable: ' . $wrapper,
        ];
    }

    $defaults = [
        'PYTHONPATH' => REPO_ROOT . '/src',
        // Override these with your actual model locations as needed.
        'DET_ENGINE' => REPO_ROOT . '/models/detector/yolov9-s_plate_fp16.engine',
        'OCR_ONNX' => REPO_ROOT . '/models/ocr/cct_s_v1_global.onnx',
        'PLATE_CONFIG' => REPO_ROOT . '/models/ocr/cct_s_v1_global_plate_config.yaml',
    ];
    $env = array_merge($defaults, $envOverrides);

    $cmdEnv = [];
    foreach ($env as $key => $value) {
        $cmdEnv[] = sprintf('%s=%s', $key, escapeshellarg($value));
    }
    if ($textOnly) {
        $cmdEnv[] = 'TEXT_ONLY=1';
    }

    $cmd = sprintf(
        '%s %s %s',
        implode(' ', $cmdEnv),
        escapeshellcmd($wrapper),
        escapeshellarg($imagePath)
    );

    $descriptorSpec = [
        1 => ['pipe', 'w'], // stdout
        2 => ['pipe', 'w'], // stderr
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
            'stdout' => $stdout,
            'stderr' => $stderr,
        ];
    }

    $payload = json_decode($stdout, true);
    if (!is_array($payload)) {
        return [
            'ok' => false,
            'exit_code' => 2,
            'stdout' => $stdout,
            'stderr' => 'failed to decode JSON output',
        ];
    }

    return [
        'ok' => true,
        'exit_code' => 0,
        'stdout' => $stdout,
        'stderr' => $stderr,
        'data' => $payload,
    ];
}

// Example: minimal handler for an uploaded file (POST form with input name "image").
if (php_sapi_name() !== 'cli' && $_SERVER['REQUEST_METHOD'] === 'POST') {
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
