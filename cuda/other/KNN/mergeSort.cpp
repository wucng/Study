// https://zhuanlan.zhihu.com/p/36075856

#include <iostream>
#include <vector>
#include <cstdlib>
using namespace std;

#define len(x) sizeof(x)/sizeof(x[0])

template <typename T>
class MergeSort {
    public:
    MergeSort(T array[],int len_arr):array(array),len_arr(len_arr){
        temp = new T[len_arr];
    }
    ~MergeSort()
    {
        delete[] temp;
    }

    void mergeSort() {
        if (array == NULL || len_arr == 0)
            return;
        mergeSort(array, 0, len_arr - 1, temp);
 
    }
    // 归并
    private:
    T *temp = NULL,*array=NULL;
    int len_arr;

    void mergeSort(T array[], int first, int last, T temp[]) {
        if (first < last) {
            int mid = (first + last) / 2;
            mergeSort(array, first, mid, temp); // 递归归并左边元素
            mergeSort(array, mid + 1, last, temp); // 递归归并右边元素
            mergeArray(array, first, mid, last, temp); // 再将二个有序数列合并
        }
    }
 
    /**
     * 合并两个有序数列
     * array[first]~array[mid]为第一组
     * array[mid+1]~array[last]为第二组
     * temp[]为存放两组比较结果的临时数组
     */
    void mergeArray(T array[], int first, int mid, int last, T temp[]) {
        int i = first, j = mid + 1; // i为第一组的起点, j为第二组的起点
        int m = mid, n = last; // m为第一组的终点, n为第二组的终点
        int k = 0; // k用于指向temp数组当前放到哪个位置
        while (i <= m && j <= n) { // 将两个有序序列循环比较, 填入数组temp
            if (array[i] <= array[j])
                temp[k++] = array[i++];
            else
                temp[k++] = array[j++];
        }
        while (i <= m) { // 如果比较完毕, 第一组还有数剩下, 则全部填入temp
            temp[k++] = array[i++];
        }
        while (j <= n) {// 如果比较完毕, 第二组还有数剩下, 则全部填入temp
            temp[k++] = array[j++];
        }
        for (i = 0; i < k; i++) {// 将排好序的数填回到array数组的对应位置
            array[first + i] = temp[i];
        }
    }

};

int main()
{   
    // vector<int> vec;
    // for(int i=0;i<10;++i)
    //     vec.push_back(rand()%10);
    int array[10];
    for(int i=0;i<10;++i)
        array[i] = rand()%10;

    for(auto c : array)
        cout<<c<<" ";
    cout << endl;

    MergeSort<int> m(array,len(array));
    m.mergeSort();

    for(auto c : array)
        cout<<c<<" ";
    cout << endl;

    return 0;
}
