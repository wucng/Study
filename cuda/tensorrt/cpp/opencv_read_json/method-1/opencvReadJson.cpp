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
    void readFile(map<string,float*>& m)
    {   
        FileStorage fs;
        fs.open(filename, FileStorage::READ);
        // 找到根节点
        FileNode node = fs["dense/w1"];
        // 根据根节点找到子节点
        int rows=(int)(node["in_channel"]);
        int cols=(int)(node["out_channel"]);
        float arr[rows*cols]={0.0f};
        FileNodeIterator it = node["value"].begin(), it_end = node["value"].end(); // Go through the node
        for (int i=0; it != it_end; ++it,++i)
            arr[i]=(float)*it;

        m.insert(pair<string,float*>("dense/w1",arr));

        node = fs["dense/b1"];
        cols=(int)(node["out_channel"]);
        float arr2[cols]={0.0f};

        it = node["value"].begin(), it_end = node["value"].end();
        for (int i=0; it != it_end; ++it,++i)
            arr2[i]=(float)*it;
        
        m.insert(pair<string,float*>("dense/b1",arr2));

        // print
        mycout<<"dense/w1"<<endl;
        for(auto c : arr)
            cout<<c<<" ";
        cout<<endl;

        mycout<<"dense/b1"<<endl;
        for(auto c : arr2)
            cout<<c<<" ";
        cout<<endl;

        fs.release(); 
    }

private:
    string filename;
};


int main(int ac, char** av)
{
    string filename = av[1];

    map<string,float*> m;

    Json json(filename);
    json.readFile(m);

    return 0;
}