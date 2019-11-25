#include <opencv2/core.hpp>
#include <iostream>
#include <string>
#include <map>
using namespace cv;
using namespace std;

#define mycout cout<<"["<<__FILE__<<":"<<__LINE__<<"]: "
#define len(x) sizeof(x)/sizeof(x[0])

template<typename T>
void pprint(T contents[],int n)
{
    mycout<<"print contents"<<endl;
    for(int i=0;i<n;i++)
        cout<<contents[i]<<",";
    cout<<endl;
}

class Json
{
public:
    Json(string filename):filename(filename){};

    // template<typename T>
    void readFile(map<string,Mat>& m)
    {   
        FileStorage fs;
        fs.open(filename, FileStorage::READ);
        // 找到根节点
        FileNode node = fs["dense/w1"];
        Mat w1;
        node >> w1; // 直接读入
        m.insert(pair<string,Mat>("dense/w1",w1));

        node = fs["dense/b1"];
        Mat b1;
        node >> b1; 
        m.insert(pair<string,Mat>("dense/b1",b1));
        
        // print
        mycout<<"dense/w1"<<endl;
        cout<<w1<<endl;

        mycout<<"dense/b1"<<endl;
        cout<<b1<<endl;

        fs.release(); 
    }

private:
    string filename;
};


int main(int ac, char** av)
{
    if(ac<2) 
    {

        mycout<<"需要传入参数：\n"<<"example: ./a.out tets.json"<<endl;
        return -1;
    }
    string filename = av[1];

    map<string,Mat> m;

    Json json(filename);
    json.readFile(m);

    return 0;
}