<?php
// Pastikan folder uploads ada
$upload_dir = __DIR__ . '/images/';
if (!file_exists($upload_dir)) {
    mkdir($upload_dir, 0777, true);
}

$iml = isset($_POST['iml']) ? ($_POST['iml']) : "0";
$nopol = isset($_POST['nopol']) ? ($_POST['nopol']) : "";
$apiUrl = getenv('ALPR_API_URL') ?: '';
$apiToken = getenv('ALPR_API_TOKEN') ?: '';
$minConf = getenv('ALPR_MIN_CONF') ?: '';

if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_FILES['image'])) {
    $file_tmp  = $_FILES['image']['tmp_name'];
    $file_name = basename($_FILES['image']['name']);
    $target    = $upload_dir . $iml . "_" . date("YmdHis") . "_" . $_FILES['image']['name'];
	
    	if ($_FILES["image"]["error"] > 0) {
		header('Content-Type: application/json');
		echo json_encode([
            'status' => 'error',
            'message' => $_FILES["image"]["error"]
        ]);
    	} else {
    	// Simpan file ke server
    	if (move_uploaded_file($file_tmp, $target)) {
            // Option A: call warm ALPR API if configured
            if ($apiUrl) {
                $ch = curl_init();
                $endpoint = rtrim($apiUrl, '/') . '/v1/alpr';
                $cfile = new CURLFile($target, mime_content_type($target) ?: 'image/jpeg', basename($target));
                $post = [ 'image' => $cfile, 'camera_id' => 'rfid-gate' ];
                if ($minConf !== '') { $post['min_conf'] = $minConf; }
                curl_setopt($ch, CURLOPT_URL, $endpoint);
                curl_setopt($ch, CURLOPT_POST, true);
                curl_setopt($ch, CURLOPT_POSTFIELDS, $post);
                curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
                curl_setopt($ch, CURLOPT_CONNECTTIMEOUT_MS, 500);
                curl_setopt($ch, CURLOPT_TIMEOUT_MS, 1500);
                $headers = [];
                // API expects X-ALPR-Token when auth is enabled
                if ($apiToken) { $headers[] = 'X-ALPR-Token: ' . $apiToken; }
                if ($headers) { curl_setopt($ch, CURLOPT_HTTPHEADER, $headers); }
                $apiResp = curl_exec($ch);
                $apiErr  = curl_error($ch);
                $status  = curl_getinfo($ch, CURLINFO_HTTP_CODE);
                curl_close($ch);
                if ($apiResp !== false && $status >= 200 && $status < 300) {
                    // Adapt API JSON {status, plates:[{text, ocr_raw, valid, ...}], ...}
                    $data = json_decode($apiResp, true);
                    $NoPolisi = '';
                    $msg = 'OK';
                    if (is_array($data)) {
                        $statusLabel = isset($data['status']) ? (string)$data['status'] : '';
                        if ($statusLabel === 'ok' && isset($data['plates']) && is_array($data['plates']) && count($data['plates']) > 0) {
                            $best = $data['plates'][0];
                            $textBest = isset($best['text']) ? (string)$best['text'] : '';
                            $textRaw = isset($best['ocr_raw']) ? (string)$best['ocr_raw'] : '';
                            $validBest = !empty($best['valid']);
                            $candidate = $textBest !== '' ? $textBest : $textRaw;
                            $NoPolisi = str_replace(' ', '', strtoupper($candidate));
                            $validRegex = preg_match('/^[A-Z]{1,2}[0-9]{1,4}[A-Z]{0,3}$/', $NoPolisi) === 1;
                            if (!$validBest && !$validRegex) {
                                $msg = 'Wrong Plate Number';
                            }
                        } else {
                            $msg = 'No plate';
                        }
                    }
                    $isOk = ($NoPolisi !== '') && (preg_match('/^[A-Z]{1,2}[0-9]{1,4}[A-Z]{0,3}$/', $NoPolisi) === 1);
                    $success = $isOk;
                    if ($isOk && $nopol !== '') {
                        if ($nopol === $NoPolisi) {
                            $msg = 'No Polisi sama';
                            $success = true;
                        } else {
                            $msg = 'No Polisi tidak sama';
                            $success = false; // preserve legacy behavior
                        }
                    }
                    $response = [
                        'success' => $success,
                        'plate'   => $NoPolisi,
                        'message' => $msg,
                    ];
                    header('Content-Type: application/json');
                    echo json_encode($response);
                    return;
                }
                // fall back to local exec path if API fails
            }
        	try{
				$projectRoot = '/home/iks-ai2/Development/ALPR_Jetson';
				$python = $projectRoot . '/venv/bin/python';
				$script = $projectRoot . '/tools/alpr_text_only.py';
				$image  = realpath($target);

				if ($image === false) {
					throw new RuntimeException('failed to resolve uploaded image path');
				}
				if (!chdir($projectRoot)) {
					throw new RuntimeException('failed to change directory to project root');
				}

				$env = [
					'DET_ENGINE'   => $projectRoot . '/models/detector/yolov9-s_plate_fp16.engine',
					'OCR_ONNX'     => $projectRoot . '/models/ocr/cct_s_v1_global.onnx',
					'PLATE_CONFIG' => $projectRoot . '/models/ocr/cct_s_v1_global_plate_config.yaml',
					'ONNX_PROVIDER'=> 'cuda',
				];

				$export = '';
				foreach ($env as $key => $value) {
					$export .= $key . '=' . escapeshellarg($value) . ' ';
				}

				$cmd = $export . escapeshellarg($python) . ' ' . escapeshellarg($script) . ' ' . escapeshellarg($image);
				$outputLines = [];
				$returnVar = 0;
				exec($cmd . ' 2>&1', $outputLines, $returnVar);
				$output = trim(implode("\n", $outputLines));

            	file_put_contents('/var/www/html/lpr/log/debug.log', "[" . date('c') . "] cmd={$cmd} rc={$returnVar}\n{$output}\n", FILE_APPEND);

				if ($returnVar === 0) {
                $NoPolisi = str_replace(" ","",$output);
                	if(!preg_match("/^[A-Z]{1,2}[0-9]{1,4}[A-Z]{0,3}$/", $NoPolisi)){
                    	$response = [
							'success' => false,
							'plate'  => $NoPolisi,
							//'code'   => $returnVar,
                        	'message' => 'Wrong Plate Number'
						];
                    }else{
                    	if($nopol == $NoPolisi){
                    		$response = [
								'success' => true,
								'plate'  => $NoPolisi,
								//'code'   => $returnVar
                        		'message' => 'No Polisi sama'
							];
                    	}else{
                    		$response = [
								'success' => false,
								'plate'  => $NoPolisi,
								//'code'   => $returnVar
                            	'message' => 'No Polisi tidak sama'
							];
                    	}                    	
                    }					
				} elseif ($returnVar === 3) {
					$response = [
						'success' => false,
						'plate'   => '',
						//'code'    => $returnVar,
						'message' => 'No plate detected or plate rejected by post-processing'
					];
				} else {
					$response = [
						'success' => false,
						'plate'   => '',
						//'code'    => $returnVar,
						'message' => $output !== '' ? $output : 'ALPR helper failed'
					];
				}

			header('Content-Type: application/json');
			echo json_encode($response);
        	}catch(Exception $e){
        		header('Content-Type: application/json');
        		echo json_encode([
        			'success' => false,
					'plate'   => '',
        			'message' => $e->getMessage()
    			]);
        	}
    	} else {
        	header('Content-Type: application/json');
        	echo json_encode([
            	'success' => false,
				'plate'   => '',
            	'message' => 'Gagal upload file. ' . $target
        	]);
    	}
	}
} else {
    header('Content-Type: application/json');
    echo json_encode([
        'success' => false,
    	'plate'   => '',
        'message' => 'Gunakan POST dengan field image.'
    ]);
}
?>
