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
