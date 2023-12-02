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
