// g++ test.cpp -lpthread -w -std=c++11
/**
thread id:140587606615808
thread id:140587606615808
cost time:0.370356 s    sum:50000005000000
*/

#include <pthread.h>
#include <atomic> // 原子变量 解决线程竞争(冲突)
#include <iostream>
#include <vector>
#include <ctime>
#include <cmath>

using namespace std;

void assignValue(vector<int>& vec,int n)
{
    for(int i=0;i<n;i++)
        vec.push_back(i+1);
}

namespace withAtomic
{
    atomic_long m_sum{ 0 };
    // atomic_int m_sum{ 0 };//不会发生线程冲突，线程安全

    typedef struct Data
    {
        vector<int> vec;
        int start_index;
        int end_index;

        Data(const vector<int> &vec,const int &start_index, const int &end_index)
        {
            this->vec=vec;
            this->start_index=start_index;
            this->end_index=end_index;
        }


    }Data;

    /**普通函数*/
    void* exec_add(void * ptr) // pthread 创建的函数必须是这种格式
    {
        cout<<"thread id:"<<pthread_self()<<endl;
        Data *data=(Data *)ptr;
        for(int i =data->start_index;i<data->end_index;i++)
        {
            // 由于多个线程同一时间都会访问m_sum，会造成线程竞争，
            // 必须加锁，保证线程同步,或者使用atomic 变量
            m_sum+=data->vec[i]; // only for atomic_int
        }
        pthread_exit(NULL); // 线程退出(只退出当前线程) 不要使用exit，会导致所有线程都退出
    }

    void do_threads(const vector<int>& vec,const size_t &NUMS_VECTOR,const size_t& NUMS_THREAD)
    {
        const size_t NUMS_VECTOR_PER_THREAD=NUMS_VECTOR/NUMS_THREAD+1;

        pthread_t  tid[NUMS_THREAD]; // 创建线程数组

        // 统计时间
        clock_t start = clock();
        for(int i=0;i<(int)NUMS_THREAD;i++)
        {
            Data data(vec,i*NUMS_VECTOR_PER_THREAD,min((i+1)*NUMS_VECTOR_PER_THREAD,(size_t)NUMS_VECTOR));
            pthread_create(&tid[i],NULL,exec_add,&data);
            pthread_join(tid[i], NULL);
        }

        clock_t finish = clock();
        cout<<"cost time:"<<(double)(finish-start)/CLOCKS_PER_SEC <<" s\t"<<
        "sum:"<<m_sum<<endl;
    }

}


namespace withoutAtomic
{
    unsigned long long m_sum = 0; // 存储最终的加和结果

    typedef struct Data
    {
        vector<int> vec;
        int start_index;
        int end_index;

        Data(const vector<int> &vec,const int &start_index, const int &end_index)
        {
            this->vec=vec;
            this->start_index=start_index;
            this->end_index=end_index;
        }
    }Data;

    /* 互斥量 */
    pthread_mutex_t m_lock;
    // pthread_mutex_init(&m_lock,NULL); // 初始化锁

    /**普通函数*/
    void* exec_add(void * ptr) // pthread 创建的函数必须是这种格式
    {
        cout<<"thread id:"<<pthread_self()<<endl;
        Data *data=(Data *)ptr;
        for(int i =data->start_index;i<data->end_index;i++)
        {
            // 由于多个线程同一时间都会访问m_sum，会造成线程竞争，
            // 必须加锁，保证线程同步,或者使用atomic 变量
            pthread_mutex_lock(&m_lock); //阻塞加锁
            m_sum+=data->vec[i];
            pthread_mutex_unlock(&m_lock); // 解锁
        }
        pthread_exit(NULL); // 线程退出(只退出当前线程) 不要使用exit，会导致所有线程都退出
    }

    void do_threads(const vector<int>& vec,const size_t &NUMS_VECTOR,const size_t& NUMS_THREAD)
    {
        const size_t NUMS_VECTOR_PER_THREAD=NUMS_VECTOR/NUMS_THREAD+1;

        pthread_t  tid[NUMS_THREAD]; // 创建线程数组

        // 统计时间
        clock_t start = clock();
        for(int i=0;i<(int)NUMS_THREAD;i++)
        {
            Data data(vec,i*NUMS_VECTOR_PER_THREAD,min((i+1)*NUMS_VECTOR_PER_THREAD,(size_t)NUMS_VECTOR));
            pthread_create(&tid[i],NULL,exec_add,&data);
            pthread_join(tid[i], NULL);
        }

        pthread_mutex_destroy(&m_lock);// 销毁锁
        clock_t finish = clock();
        cout<<"cost time:"<<(double)(finish-start)/CLOCKS_PER_SEC <<" s\t"<<
        "sum:"<<m_sum<<endl;
    }

}

int main()
{
    const unsigned int NUMS_VECTOR=10000000;
    const size_t NUMS_THREAD=2; // 创建的线程数

    // 创造容器
    vector<int> vec;
    vec.reserve(NUMS_VECTOR);  // 提前分配空间 ，也可以让其自动分配（不加这行）
    // 赋值
    assignValue(vec,NUMS_VECTOR);

    // 使用原子变量 保证线程同步
    // withAtomic::do_threads(vec,NUMS_VECTOR,NUMS_THREAD);

    // 使用线程锁 同步线程
    pthread_mutex_init(&(withoutAtomic::m_lock),NULL); // 初始化锁(必须放在main函数中初始化)
    withoutAtomic::do_threads(vec,NUMS_VECTOR,NUMS_THREAD);

    return 0;
}
