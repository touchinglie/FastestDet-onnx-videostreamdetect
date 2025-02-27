g++ -o FastestStreamDet FastestStreamDet.cpp -I /usr/local/include/ncnn /usr/local/lib/ncnn/libncnn.a `pkg-config --libs --cflags opencv4` -fopenmp
