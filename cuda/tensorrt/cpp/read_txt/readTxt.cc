#include <iostream>
#include <fstream>
//#include <ostream>
//#include <cstdio>
//#include <cstdlib>
#include <string>
#include <string.h>
#include <cassert>
#include <map>

using namespace std;
//using std::cout;
//using std::endl;

#define mycout (cout<<"["<<__FILE__<<":"<<__LINE__<<"]: "<<endl) // 下划线

bool strInstr(string s1,string s2)
{
    return s2.find(s1)!=-1;
}

void str2arr(const string &str,float* arr,int n)
{
    int start_index=0,end_index=0;
    int m=0;
    for(int i=0;i<str.size();++i)
    {
        if(str[i]==','|| i==str.size()-1)
        {
            end_index=i;
            arr[m]=stof(str.substr(start_index,end_index-start_index));
            start_index=end_index+2;
            m++;
        }
    }
    assert(m=n);
}

void printMap(map<string,float*>&m)
{
    for(auto c:m)
        cout<<"key = "<<c.first<<" value = "<<c.second[0]<<endl;
    cout<<endl;
}

int getWeightsByNmae(const string &name,ifstream &infile,float **tmp,int &rows,int &cols)
{
    float *arr;
    string buf;
    while(getline(infile,buf)) // 不会读入换行符
    {
        //cout<<buf<<endl;
        if(name==buf)
        {
            // buf.copy(name_str,buf.size());
            // name_str=buf;
            getline(infile,buf);
            rows=stoi(buf.substr(5));
            getline(infile,buf);
            cols=stoi(buf.substr(5));

            // 分配空间
            arr=new float[rows*cols];
            // float arr[rows*cols]={0.0f};

            // 读入数组
            getline(infile,buf);

            str2arr(buf.substr(6,buf.size()-7),arr,rows*cols);

            // 写入map
            // dict.insert(pair<string,float*>(name_str,arr));

            // free
            //delete[] arr;

            *tmp=arr;
            return 0;
        }
    }

    return -1;
}

int main(int argc,char* argv[])
{
    if(argc<2)
    {
        mycout<<"参数不足"<<endl;
        return -1;
    }

    // 读文件
    ifstream infile;
    infile.open(argv[1]);
    if(!infile)
    {
        mycout<<"open file failed" << endl;
        return -1;
    }


    float *arr;
    int rows=0,cols=0;
    getWeightsByNmae("dense/w1",infile,&arr,rows,cols);
    for(int i=0;i<rows*cols;++i)
        cout<<arr[i]<<" ";
    cout<<endl;

    infile.close();

    //打印信息
    // printMap(dict);

    return 0;
}
