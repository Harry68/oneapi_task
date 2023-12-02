# OneAPI作业题解
# 作业一：并行矩阵乘法 题解
## 题目描述
编写一个基于 oneAPI 的 C++/SYCL 程序来执行矩阵乘法操作。需要考虑大尺寸矩阵的乘法操作以及不同线程之间的数据依赖关系。通常在实现矩阵乘法时，可以使用块矩阵乘法以及共享内存来提高计算效率。
## 题目分析
利用基于SYCL的编程模型在GPU上实现矩阵乘法的计算，步骤如下：
1. 分配内存：在主机端分配内存空间用于存储输⼊矩阵和输出矩阵，同时在GPU端分配内存空间用于存储相应的输入和输出数据。
2. 数据传输：将输入矩阵数据从主机端内存传输到GPU端内存中。
3. 核函数调用：在GPU上使用核函数来实现并行计算，核函数会分配线程块和线程来处理不同的数据块。
4. 并行计算：在核函数中，每个线程负责计算输出矩阵的⼀个单独的元素。
5. 数据传输：计算完成后，将输出矩阵数据从GPU端内存传输回主机端内存中，以便进⼀步处理或分析。
## 实现方案
程序使用 SYCL（Single-source Heterogeneous C++ Code）来实现矩阵乘法操作，主要包含以下函数：
1. **矩阵乘法函数 `matrixMultiply`**:
   - 创建 SYCL 缓冲区（buffer）用于存储输入矩阵 A、B 和输出矩阵 C。
   - 通过 SYCL 访问器（accessor）获取对缓冲区的读写权限。
   - 使用 `parallel_for` 在不同的工作项上并行执行矩阵乘法操作。
   - 通过数据依赖关系，确保正确的计算顺序。

2. **分块矩阵乘法函数 `matrix_multiply_block`**:
   - 在矩阵乘法的基础上，引入块矩阵乘法的思想，将矩阵划分为小块。
   - 创建共享内存区域，将矩阵 B 的一部分加载到共享内存中，以提高访存效率。
   - 使用 barrier 函数等待所有工作项完成共享内存的加载。
   - 在块内执行矩阵乘法，再次使用 barrier 函数等待所有工作项完成计算。

3. **输出结果函数 `print`**:
   - 将结果以矩阵形式输出。

4. **主函数 `main`**:
   - 读取输入矩阵 A 和 B 的值。
   - 创建 SYCL 队列，选择 GPU 设备进行计算。
   - 调用矩阵乘法函数进行计算。
   - 输出结果矩阵 C。
## 具体代码
```cpp
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
```
# 作业二：并行排序算法 题解
## 题目描述
使用基于 oneAPI 的 C++/SYCL 实现一个高效的并行归并排序。需要考虑数据的分割和合并以及线程之间的协作。
## 题目分析
归并排序是⼀种分治算法，其基本原理是将待排序的数组分成两部分，分别对这两部分进行排序，然后将已排序的子数组合并为⼀个有序数组。可考虑利用了异构并行计算的特点，将排序和合并操作分配给多个线程同时执行，以提高排序效率。具体实现过程如下：
1. 将待排序的数组分割成多个较小的子数组，并将这些⼦数组分配给不同的线程块进行处理。
2. 每个线程块内部的线程协作完成子数组的局部排序。
3. 通过多次迭代，不断合并相邻的有序⼦数组，直到整个数组有序。
## 实现方案
程序使用 SYCL（Single-source Heterogeneous C++ Code）来实现并行归并排序，主要包含以下函数：

1. **读取数据函数 `read_data`**:
   - 从文件中读取整数数据，存储在向量中。

2. **合并有序数组函数 `merge`**:
   - 创建 SYCL 缓冲区（buffer）用于存储数据。
   - 通过 SYCL 访问器（accessor）获取对缓冲区的读写权限。
   - 使用 `parallel_for` 在不同的工作项上并行执行合并有序数组操作。
   - 在合并过程中，创建一个临时数组用于存储合并后的有序数据。

3. **归并排序函数 `mergeSort`**:
   - 递归地将数组分割为较小的部分，然后调用合并函数进行合并。
   - 在分割和合并的过程中，通过 SYCL 并行框架实现高效的归并排序。

4. **输出结果函数 `print`**:
   - 将排序结果以序列形式输出。

5. **主函数 `main`**:
   - 从文件中读取数据，创建 SYCL 队列和缓冲区。
   - 调用归并排序函数进行并行排序。
   - 输出排序结果。
## 具体代码
```cpp
#include <CL/sycl.hpp>
#include <iostream>
#include <fstream>
#include <vector>

// 从文件中读取数据
void read_data(std::string& filename, std::vector<int>& data) {
    std::ifstream file(filename);
    int num;
    while (file >> num) data.push_back(num);

    file.close();
}

// 合并有序数组
void merge(sycl::queue& queue, sycl::buffer<int, 1>& buffer, int left, int mid, int right) {
    sycl::range<1> globalSize{right - left + 1};

    queue.submit([&](sycl::handler& cgh) {
        // 获取读写访问权限
        auto accessor = buffer.get_access<sycl::access::mode::read_write>(cgh);

        // 合并有序数组
        cgh.parallel_for<class merge_kernel>(globalSize, [=](sycl::id<1> idx) {
            int i = left + idx[0], j = mid + 1 + idx[0];
            if (j <= right) {
                int a = i, b = j, n = 0;
                int* temp = new int[right - left + 1];

                while (a <= mid && b <= right) {
                    if (accessor[a] <= accessor[b]) temp[n ++] = accessor[a ++];
                    else temp[n ++] = accessor[b ++];
                }

                while (a <= mid) temp[n ++] = accessor[a ++];
                while (b <= right) temp[n ++] = accessor[b ++];

                for (int p = 0; p < n; p ++) accessor[i + p] = temp[p];
                delete[] temp;
            }
        });
    });

    queue.wait();
}

// 归并排序
void mergeSort(sycl::queue& queue, sycl::buffer<int, 1>& buffer, int left, int right) {
    if (left >= right) return;
    int mid = left + right >> 1;
    mergeSort(queue, buffer, left, mid);
    mergeSort(queue, buffer, mid + 1, right);
    merge(queue, buffer, left, mid, right);
}

// 输出结果
void print(std::vector<int> data) {
    for (int i = 0; i < n; i ++) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::string filename = "input.txt";
    std::vector<int> data = read_data(filename); // 从文件中读取数据
    int n = data.size();

    // 创建SYCL队列和缓冲区
    sycl::queue queue(sycl::gpu_selector{});
    sycl::buffer<int, 1> buffer(data.data(), sycl::range<1>(SIZE));

    // 执行归并排序
    mergeSort(queue, buffer, 0, n - 1);
    std::vector<int> result = buffer.get_access<sycl::access::mode::read>();

    // 输出结果
    print(matrixC)

    return 0;
}
```
# 作业三：图像卷积并行加速 题解
## 题目描述
使用基于 oneAPI 的 C++/SYCL 实现一个用于计算图像的卷积操作。输入为一个图像矩阵和一个卷积核矩阵，输出为卷积后的图像。

## 题目分析
图像卷积是一种常见的图像处理操作，用于应用各种滤波器和特征检测器。其原理可以简单地描述为在图像的每个像素上应用一个小的矩阵（通常称为卷积核或滤波器），并将卷积核中的元素与图像中对应位置的像素值相乘，然后将所有乘积的和作为结果。这个过程可以看作是对图像进行了平滑、锐化、边缘检测等操作。假设有⼀个大小为$M × N$的输入图像I和一个大小为$m × n$的卷积核 $K$ 。图像卷积操作可以用下面的数学公式来表示：
$$
S(i,j)=\sum\limits_{k}\sum\limits_{l}I(i+k,j+l)\cdot K(k,l)
$$其中$S (i , j )$是卷积操作的结果图像中位置$(i , j )$处的像素值。$I (i + k , j + l )$是图像中位置$(i + k , j + l )$处的像素值，$K (k , l )$是卷积核中位置$(k , l )$处的权重。卷积核通常是一个小的⼆维矩阵，用于捕捉图像中的特定特征。在异构计算编程中，可以使用并行计算来加速图像卷积操作。通过将图像分割成小块，然后在GPU上并行处理这些块，可以实现高效的图像卷积计算。通过合理的块大小和线程组织方式，可以最大限度地利用GPU的并行计算能力来加速图像处理过程。
## 实现方案
程序使用 SYCL（Single-source Heterogeneous C++ Code）来实现图像卷积操作，主要包含以下函数：

1. **读取数据函数 `read_data`**:
   - 从文件中读取整数数据，存储在二维向量中，分别表示图像矩阵和卷积核矩阵。

2. **卷积操作函数 `convolution`**:
   - 创建 SYCL 缓冲区（buffer）用于存储输入图像、卷积核和输出图像。
   - 使用 `parallel_for` 在不同的工作项上并行执行卷积操作。
   - 通过获取读写访问权限，实现对输入图像和卷积核的读取，对输出图像的写入。
   - 计算卷积操作的累加和，并写入输出图像。

3. **输出结果函数 `print`**:
   - 将卷积后的图像以矩阵形式输出。

4. **主函数 `main`**:
   - 从文件中读取图像和卷积核的数据。
   - 获取图像和卷积核的大小。
   - 创建 SYCL 队列和缓冲区。
   - 调用卷积操作函数进行并行计算。
   - 输出卷积结果。
## 具体代码
```cpp
#include <CL/sycl.hpp>
#include <iostream>
#include <fstream>
#include <vector>

// 从文件中读取数据
void read_data(std::string& filename, std::vector<std::vector<int>>& data) {
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::vector<int> row;
        std::istringstream sstream(line);
        int num;
        while (sstream >> num) row.push_back(num);
        data.push_back(row);
    }

    file.close();
}

// 卷积操作
void convolution(sycl::queue& queue, sycl::buffer<int, 2>& inputBuffer, sycl::buffer<int, 2>& kernelBuffer, sycl::buffer<int, 2>& outputBuffer, int imageWidth, int imageHeight, int kernelSize) {
    int outputHeight = imageHeight - kernelSize + 1; 
    int outputWidth = imageWidth - kernelSize + 1;
    sycl::range<2> globalSize{outputWidth, outputHeight};

    queue.submit([&](sycl::handler& cgh) {
        // 获取读写访问权限
        auto inputAccessor = inputBuffer.get_access<sycl::access::mode::read>(cgh);
        auto kernelAccessor = kernelBuffer.get_access<sycl::access::mode::read>(cgh);
        auto outputAccessor = outputBuffer.get_access<sycl::access::mode::write>(cgh);

        // 卷积操作
        cgh.parallel_for<class convolution_kernel>(globalSize, [=](sycl::item<2> item) {
            int i = item.get_global_id(0), j = item.get_global_id(1);
            int sum = 0;
            for (int n = 0; n < kernelSize; n ++) {
                for (int m = 0; m < kernelSize; m ++) {
                    int x = i + n, y = j + m;
                    sum += inputAccessor[x][y] * kernelAccessor[n][m];
                }
            }
            outputAccessor[i][j] = sum;
        });
    });

    queue.wait();
}

// 输出结果
void print(std::vector<int>& data) {
    for (int i = 0; i < data.size(); i ++) {
        for (int j = 0; j < data[0].size(); j ++) {
            std::cout << data[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // 从文件中读取图像和卷积核的数据
    std::vector<std::vector<int>> image = read_data("image.txt");
    std::vector<std::vector<int>> kernel = read_data("kernel.txt");

    // 获取图像和卷积核的大小
    int imageHeight = image.size();
    int imageWidth = image[0].size();
    int kernelSize = kernel.size();

    // 卷积结果的大小
    int resultHeight = imageHeight - kernelSize + 1; 
    int resultWidth = imageWidth - kernelSize + 1;

    // 创建SYCL队列和缓冲区
    sycl::queue queue(sycl::gpu_selector{});
    sycl::buffer<int, 2> inputBuffer(image.data(), sycl::range<2>(imageHeight, imageWidth));
    sycl::buffer<int, 2> kernelBuffer(kernel.data(), sycl::range<2>(kernelSize, kernelSize));
    sycl::buffer<int, 2> outputBuffer(sycl::range<2>(resultHeight, resultWidth));

    // 卷积操作
    convolution(queue, inputBuffer, kernelBuffer, outputBuffer, imageWidth, imageHeight, kernelSize);
    std::vector<std::vector<int>> result = outputBuffer.get_access<sycl::access::mode::read>();

    // 输出结果
    print(result);

    return 0;
}
```

# 个人收获
通过使用oneAPI的C++/SYCL实现并行矩阵乘法、并行归并排序和并行图像卷积，我深入了解了异构计算环境和数据并行性和oneAPI编程模型，并学到了有效的内存管理。通过实践这些并行算法，提升了我的并行编程技能，为处理大规模数据和高性能计算任务提供了更为丰富的经验和工具。这些收获将对我的未来工作在并行计算领域中发挥积极作用。
