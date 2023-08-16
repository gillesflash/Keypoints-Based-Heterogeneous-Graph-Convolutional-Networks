This keypoint detection model is based on mask RCNN and keypoint RCNN

address for keypoint detection model:


	--test_dumper.py(code)
	--test_excavator.py(code)
	--test_dozer.py(code)
	--DT_dozer.txt(keypoints out put)
	--DT_dumper.txt
	--DT_excvator.txt
	--model
	  --weights
	    --keypointsrcnn_weights_dozer0.pth
	    --keypointsrcnn_weights_dumper0.pth
	    --keypointsrcnn_weights_excavator0.pth
	--dataset0
	  --DT_dozer
	    --train
	      --images(input image)
	  --DT_dumper
	    --train
	      --images
	  --DT_excavator
	    --train
	      --images
