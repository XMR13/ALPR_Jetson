<?php
// Pastikan folder uploads ada
$upload_dir = __DIR__ . '/images/';
if (!file_exists($upload_dir)) {
    mkdir($upload_dir, 0777, true);
}

$iml = isset($_POST['iml']) ? ($_POST['iml']) : "0";
$nopol = isset($_POST['nopol']) ? ($_POST['nopol']) : "";

if ($_SERVER['REQUEST_METHOD'] == 'POST' && isset($_FILES['image'])) {
    $file_tmp  = $_FILES['image']['tmp_name'];
    $file_name = basename($_FILES['image']['name']);
    $target    = $upload_dir . $iml . "_" . date("YmdHis") . "_" . $_FILES['image']['name'];
	
	if ($_FILES["image"]["error"] > 0) {
		echo json_encode([
            'status' => 'error',
            'message' => $_FILES["file"]["error"]
        ]);
	} else {
    	// Simpan file ke server
    	if (move_uploaded_file($file_tmp, $target)) {
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

        		echo json_encode($response);
        	}catch(Exception $e){
        		echo json_encode([
        			'success' => false,
					'plate'   => '',
        			'message' => $e->getMessage()
    			]);
        	}
    	} else {
        	echo json_encode([
            	'success' => false,
				'plate'   => '',
            	'message' => 'Gagal upload file. ' . $target
        	]);
    	}
	}
} else {
    echo json_encode([
        'success' => false,
    	'plate'   => '',
        'message' => 'Gunakan POST dengan field image.'
    ]);
}   
?>