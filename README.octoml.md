## Build Docker Images via

docker build -t octoml/olive:cpu -f Dockerfile.cpu https://github.com/gaziqbal/onnx-golive.git
docker build -t octoml/olive:cuda -f Dockerfile.cuda https://github.com/gaziqbal/onnx-golive.git

## Run Docker Images

docker run -it --rm octoml/olive:cpu
docker run -it --rm --gpus all octoml/olive:cuda

## Run Benchmarking

Run a benchmarking job using the CPU image

`
docker run -it --rm -v /home/giqbal/Source/onnx-octomizer/models:/models octoml/olive:cpu optimize --model_path /models/yolov5s.onnx --result_path /models/yolov5s --providers cpu,openvino --ort_opt_level_list disable,all --run_all
`
