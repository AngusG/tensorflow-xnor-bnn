all:
	nvcc -std=c++11 -c -o obj/xnor_gemm_kernel.cu.o kernels/xnor_gemm_kernel.cu.cc -I ${TF_INC} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr --Wno-deprecated-gpu-targets
	g++ -std=c++11 -shared -o libs/xnor_gemm_kernel.so kernels/xnor_gemm_kernel.cc obj/xnor_gemm_kernel.cu.o -I ${TF_INC} -fPIC -lcudart -L ${CUDA_ROOT}/lib64

clean:
	rm obj/*.o
