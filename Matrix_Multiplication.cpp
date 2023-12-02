#include <CL/sycl.hpp>
#include <fstream>
#include <iostream>
#include <vector>

const int ROWA = 66;
const int COLA = 7;
const int ROWB = 7;
const int COLB = 31;

// 从文件中读取矩阵的值
void read_matrix(std::string& filename, std::vector<float>& matrix, int row, int col) {
    std::ifstream file(filename);
    for (int i = 0; i < row; i ++) {
        for (int j = 0; j < col; j ++) {
            file >> matrix[i * row + j];
        }
    }

    file.close();
}

// 矩阵乘法
void matrixMultiply(sycl::queue& queue, std::vector<float>& A, std::vector<float>& B, std::vector<float>& C) {
    sycl::range<2> globalSize{ROWA, COLB}; 

    // 创建缓冲区
    sycl::buffer<float, 2> bufferA(A.data(), sycl::range<2>(ROWA, COLA));
    sycl::buffer<float, 2> bufferB(B.data(), sycl::range<2>(ROWB, COLB));
    sycl::buffer<float, 2> bufferC(C.data(), sycl::range<2>(ROWA, COLB));

    queue.submit([&](sycl::handler& cgh) {
        // 获取读写访问权限
        auto accessorA = bufferA.get_access<sycl::access::mode::read>(cgh);
        auto accessorB = bufferB.get_access<sycl::access::mode::read>(cgh);
        auto accessorC = bufferC.get_access<sycl::access::mode::write>(cgh);

        // 矩阵乘法
        cgh.parallel_for<class matrix_mult>(globalSize, [=](sycl::id<2> idx) {
            int i = idx[0], j = idx[1];
            float sum = 0.0f;
            for (int k = 0; k < COLA; k ++) {
                sum += accessorA[{i, k}] * accessorB[{k, j}];
            }
            accessorC[idx] = sum;
        });
    });

    queue.wait(); // 等待队列执行完毕
}

// 分块矩阵乘法
void matrix_multiply_block(sycl::queue& queue, std::vector<float>& A, std::vector<float>& B, std::vector<float>& C) {
    int BLOCK_SIZE = 4;
    sycl::range<2> globalSize{ROWA, COLB};
    sycl::range<2> localSize{BLOCK_SIZE, BLOCK_SIZE};

    // 创建缓冲区
    sycl::buffer<float, 2> bufferA(A.data(), sycl::range<2>(MATRIX_SIZE, MATRIX_SIZE));
    sycl::buffer<float, 2> bufferB(B.data(), sycl::range<2>(MATRIX_SIZE, MATRIX_SIZE));
    sycl::buffer<float, 2> bufferC(C.data(), sycl::range<2>(MATRIX_SIZE, MATRIX_SIZE));

    queue.submit([&](sycl::handler& cgh) {
        // 获取读写访问权限
        auto accessorA = bufferA.get_access<sycl::access::mode::read>(cgh);
        auto accessorB = bufferB.get_access<sycl::access::mode::read>(cgh);
        auto accessorC = bufferC.get_access<sycl::access::mode::write>(cgh);

        // 创建BLOCK_SIZE x BLOCK_SIZE的共享内存区域
        sycl::accessor<float, 2, sycl::access::mode::read_write, sycl::access::target::local> localAcc(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), cgh);

        cgh.parallel_for<class matrix_mult_block>(sycl::nd_range<2>(globalSize, localSize), [=](sycl::nd_item<2> item) {
            int i = item.get_global_id(0), j = item.get_global_id(1), k;
            float sum = 0.0f;
            for (k = 0; k < COLA; k += BLOCK_SIZE) {
                localAcc[item.get_local_id()] = accessorB[{k + item.get_global_id(0), j}]; // 将矩阵B的一部分加载到共享内存
                item.barrier(sycl::access::fence_space::local_space); // 等待所有工作项完成共享内存的加载

                for (int u = 0; u < BLOCK_SIZE && k + u < COLA; u ++) {
                    sum += accessorA[{i, k + u}] * localAcc[{u, item.get_local_id(1)}]; // 分块乘法
                }

                item.barrier(sycl::access::fence_space::local_space); // 等待所有工作项完成分块乘法
            }

            accessorC[{i, j}] = sum; // 将乘法结果写入矩阵C
        });
    });

    queue.wait(); // 等待队列执行完毕
}


// 输出结果
void print(std::vector<float>& C) {
    for (int i = 0; i < ROWA; i ++) {
        for (int j = 0; j < COLB; j ++) {
            std::cout << C[i * ROWA + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    std::vector<float> matrixA(ROWA * COLA, 0.0);
    std::vector<float> matrixB(ROWB * COLB, 0.0);
    std::vector<float> matrixC(ROWA * COLB, 0.0);

    read_matrix("matrixA.txt", matrixA, ROWA, COLA);
    read_matrix("matrixB.txt", matrixB, ROWB, COLB);

    // 创建SYCL队列
    sycl::queue queue(sycl::gpu_selector{});

    // 执行矩阵乘法
    matrix_multiply(queue, matrixA, matrixB, matrixC);

    // 输出结果
    print(matrixC)

    return 0;
}
