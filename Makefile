all:
	nvcc -std=c++11 -c -o obj/xnor_gemm_kernel.cu.o src/xnor_gemm_kernel.cu.cc -I ${TF_INC} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr --Wno-deprecated-gpu-targets
	nvcc -std=c++11 -c -o obj/base_gemm_kernel.cu.o src/base_gemm_kernel.cu.cc -I ${TF_INC} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr --Wno-deprecated-gpu-targets
	nvcc -std=c++11 -c -o obj/concatenate_kernel.cu.o src/concatenate_kernel.cu.cc -I ${TF_INC} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr --Wno-deprecated-gpu-targets
	g++ -std=c++11 -shared -o libs/gemm_op.so src/gemm_op.cc -D_GLIBCXX_USE_CXX11_ABI=0 obj/xnor_gemm_kernel.cu.o obj/base_gemm_kernel.cu.o obj/concatenate_kernel.cu.o -I ${TF_INC} -fPIC -lcudart -L ${CUDA_ROOT}/lib64

cpp:	
	g++ -std=c++11 -shared -o libs/gemm_op.so src/gemm_op.cc -D_GLIBCXX_USE_CXX11_ABI=0 obj/xnor_gemm_kernel.cu.o obj/base_gemm_kernel.cu.o obj/concatenate_kernel.cu.o -I ${TF_INC} -fPIC -lcudart -L ${CUDA_ROOT}/lib64

clean:
	rm obj/*.o libs/gemm_op.so
