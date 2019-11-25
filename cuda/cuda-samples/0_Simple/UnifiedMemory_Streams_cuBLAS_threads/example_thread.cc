// g++ test.cpp -lpthread -std=c++11 -w
/**
CPU CORES:4
thread id:140213296207616
thread id:140213296207616
cost time:0.2144 s      sum:50000005000000
*/
#include <thread>
#include <mutex>  // for 线程锁
#include<atomic>   // 原子变量 解决线程竞争(冲突)
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

    /**普通函数*/
    void exec_add(const vector<int>& vec,const int &start_index,const int &end_index)
    {
        cout<<"thread id:"<<this_thread::get_id()<<endl;
        for(int i =start_index;i<end_index;i++)
        {
            // 由于多个线程同一时间都会访问m_sum，会造成线程竞争，
            // 必须加锁，保证线程同步,或者使用atomic 变量
            m_sum+=vec[i]; // only for atomic_int
        }
        return; // 线程退出(只退出当前线程) 不要使用exit，会导致所有线程都退出
    }

    /**仿函数*/
    class ExecAdd
    {
        public:
            void operator()(const vector<int>& vec,const int &start_index,const int &end_index)
            {
                cout<<"thread id:"<<this_thread::get_id()<<endl;
                for(int i =start_index;i<end_index;i++)
                {
                    // 由于多个线程同一时间都会访问m_sum，会造成线程竞争，
                    // 必须加锁，保证线程同步,或者使用atomic 变量
                    m_sum+=vec[i]; // only for atomic_int
                }
                return; // 线程退出(只退出当前线程) 不要使用exit，会导致所有线程都退出
            }
    };


    /**lambda表达式*/
    auto exec_add_lambda=[](const vector<int>& vec,const int &start_index,const int &end_index)->void{
        cout<<"thread id:"<<this_thread::get_id()<<endl;
        for(int i =start_index;i<end_index;i++)
        {
            // 由于多个线程同一时间都会访问m_sum，会造成线程竞争，
            // 必须加锁，保证线程同步,或者使用atomic 变量
            m_sum+=vec[i]; // only for atomic_int
        }
        return; // 线程退出(只退出当前线程) 不要使用exit，会导致所有线程都退出
    };


    void do_threads(const vector<int>& vec,const size_t &NUMS_VECTOR,const size_t& NUMS_THREAD)
    {
        const size_t NUMS_VECTOR_PER_THREAD=NUMS_VECTOR/NUMS_THREAD+1;

        thread t[NUMS_THREAD]; // 创建2个线程数组

        // 统计时间
        clock_t start = clock();
        for(int i=0;i<(int)NUMS_THREAD;i++)
        {
            /**普通函数*/
            // t[i]=thread(exec_add,vec,i*NUMS_VECTOR_PER_THREAD,min((i+1)*NUMS_VECTOR_PER_THREAD,(size_t)NUMS_VECTOR));
            /**仿函数 std::ref()*/
            // t[i]=thread(ExecAdd(),vec,i*NUMS_VECTOR_PER_THREAD,min((i+1)*NUMS_VECTOR_PER_THREAD,(size_t)NUMS_VECTOR));

            /**lambda表达式*/
            t[i]=thread(exec_add_lambda,vec,i*NUMS_VECTOR_PER_THREAD,min((i+1)*NUMS_VECTOR_PER_THREAD,(size_t)NUMS_VECTOR));

            if(t[i].joinable()) t[i].join();
        }

        clock_t finish = clock();
        cout<<"cost time:"<<(double)(finish-start)/CLOCKS_PER_SEC <<" s\t"<<
        "sum:"<<m_sum<<endl;
    }

}

namespace withoutAtomic
{
    unsigned long long m_sum = 0; // 存储最终的加和结果
    mutex g_display_mutex;

    void exec_add(const vector<int>& vec,const int &start_index,const int &end_index)
    {
        cout<<"thread id:"<<this_thread::get_id()<<endl;
        for(int i =start_index;i<end_index;i++)
        {
            // 由于多个线程同一时间都会访问m_sum，会造成线程竞争，必须加锁，保证线程同步
           lock_guard<mutex> guard(g_display_mutex); // 推荐使用这种方式
           m_sum+=vec[i];

           // or
           /*
           g_display_mutex.lock();
           m_sum+=vec[i];
           g_display_mutex.unlock();
            */
        }
        return; // 线程退出(只退出当前线程) 不要使用exit，会导致所有线程都退出
    }

    void do_threads(const vector<int>& vec,const size_t &NUMS_VECTOR,const size_t& NUMS_THREAD)
    {
        const size_t NUMS_VECTOR_PER_THREAD=NUMS_VECTOR/NUMS_THREAD+1;

        thread t[NUMS_THREAD]; // 创建2个线程数组

        // 统计时间
        clock_t start = clock();
        for(int i=0;i<(int)NUMS_THREAD;i++)
        {
            t[i]=thread(exec_add,vec,i*NUMS_VECTOR_PER_THREAD,min((i+1)*NUMS_VECTOR_PER_THREAD,(size_t)NUMS_VECTOR));
            if(t[i].joinable()) t[i].join();
        }

        clock_t finish = clock();
        cout<<"cost time:"<<(double)(finish-start)/CLOCKS_PER_SEC <<" s\t"<<
        "sum:"<<m_sum<<endl;
    }
}

int main()
{
    cout<<"CPU CORES:"<<thread::hardware_concurrency()<<endl;

    const unsigned int NUMS_VECTOR=10000000;
    const size_t NUMS_THREAD=2; // 创建的线程数

    // 创造容器
    vector<int> vec;
    vec.reserve(NUMS_VECTOR);  // 提前分配空间 ，也可以让其自动分配（不加这行）
    // 赋值
    assignValue(vec,NUMS_VECTOR);

    // 使用原子变量 保证线程同步
    withAtomic::do_threads(vec,NUMS_VECTOR,NUMS_THREAD);

    // 使用线程锁 同步线程
    // withoutAtomic::do_threads(vec,NUMS_VECTOR,NUMS_THREAD);


    return 0;
}
