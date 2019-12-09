# Caffe-Int8-To-NCNN

This tools is base on TensorRT 2.0 Int8 calibration tools,which use the KL algorithm to find the suitable threshold to quantize the activions from Float32 to Int8(-127 - 127).

This code modify  base on [https://github.com/BUG1989/caffe-int8-convert-tools] 


## Reference

For details, please read the following PDF:

[8-bit Inference with TensorRT](http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf) 

MXNet quantization implement:

[Quantization module for generating quantized (INT8) models from FP32 models](https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py)

An introduction to the principles of a Chinese blog written by my friend([bruce.zhang](https://github.com/bigbigzxl)):

[The implement of Int8 quantize base on TensorRT](https://zhuanlan.zhihu.com/zhangxiaolongOptimization)

## HowTo

The purpose of this tool(caffe-int8-to-ncnn.py)  is to save the caffemodel as an int8 ncnn model and deploy it to ncnn?

This format is already supported in the [ncnn](https://github.com/Tencent/ncnn) latest version.

```
python caffe-int8-convert-tool-dev-weight.py -h
usage: caffe-int8-convert-tool-dev-weight.py [-h] [--proto PROTO] [--model MODEL]
                                  [--mean MEAN MEAN MEAN] [--norm NORM]
                                  [--images IMAGES] [--output OUTPUT]
                                  [--group GROUP] [--gpu GPU]

find the pretrained caffemodel int8 quantize scale value

optional arguments:
  -h, --help            	show this help message and exit
  --proto PROTO         	path to deploy prototxt.
  --model MODEL         	path to pretrained caffemodel
  --mean MEAN           	value of mean
  --norm NORM           	value of normalize(scale value or std value)
  --images IMAGES       	path to calibration images
  --output_param OUTPUT_PARAM     path to output ncnn param file
  --output_bin OUTPUT_BIN       path to output ncnn bin file
  --group GROUP         enable the group scale(0:disable,1:enable,default:1)
  --gpu GPU             use gpu to forward(0:disable,1:enable,default:0)
python caffe-int8-convert-tool-dev-weight.py --proto=test/models/mobilenet_v1.prototxt --model=test/models/mobilenet_v1.caffemodel --mean 103.94 116.78 123.68 --norm=0.017 --images=test/images/ output_param=pnet.param output_param=pnet.bin --group=1 --gpu=1
```
## License

BSD 3 Clause
