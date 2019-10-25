#include <opencv2/core.hpp>
#include <iostream>
#include <string>
using namespace cv;
using namespace std;

#define mycout cout<<"["<<__FILE__<<":"<<__LINE__<<"] "
#define len(x) sizeof(x)/sizeof(x[0])

template<typename T>
void pprint(T contents[],int n)
{
    for(int i=0;i<n;i++)
        mycout<<contents[i]<<",";
    cout<<endl;
}

static void help(char** av)
{
    cout << endl
        << av[0] << " shows the usage of the OpenCV serialization functionality."         << endl
        << "usage: "                                                                      << endl
        <<  av[0] << " outputfile.yml.gz"                                                 << endl
        << "The output file may be either XML (xml) or YAML (yml/yaml). You can even compress it by "
        << "specifying this in its extension like xml.gz yaml.gz etc... "                  << endl
        << "With FileStorage you can serialize objects in OpenCV by using the << and >> operators" << endl
        << "For example: - create a class and have it serialized"                         << endl
        << "             - use it to read and write matrices."                            << endl;
}


class Json
{
public:
    Json(string myfilename):filename(myfilename){}

    template<typename T>
    void writeFile(string nodes[],int nodesLen,string childNodes[],int chnodesLen,T contents[],int contLen)
    {
        CV_Assert(nodesLen*chnodesLen == contLen);
        FileStorage fs(filename, FileStorage::WRITE);
        // write
        for(int i=0;i<nodesLen;i++)
        {
            fs << nodes[i]; // 先写父节点
            fs <<"{";
            for(int j=0;j<chnodesLen;j++)// 写子节点
            {
                fs<<childNodes[j] << contents[j+chnodesLen*i];
            }
            fs << "}";
        }
        fs.release(); // 关闭文件
    }

    template<typename T>
    void readFile(string nodes[],int nodesLen,string childNodes[],int chnodesLen,T contents[],int contLen)
    {
        CV_Assert(nodesLen*chnodesLen == contLen);
        FileNode n;
        FileStorage fs(filename, FileStorage::READ);
        // read
        for(int i=0;i<nodesLen;i++)
        {
            n = fs[nodes[i]];
            for(int j=0;j<chnodesLen;j++)
            {
                contents[j+i*chnodesLen]=(T)(n[childNodes[j]]);
            }
        }
        fs.release(); // 关闭文件
    }

private:
    string filename;

};

int main(int ac, char** av)
{
    if (ac != 2)
    {
        help(av);
        return 1;
    }
    string filename = av[1];
    /*
    { //write
        FileStorage fs(filename, FileStorage::WRITE);
        fs << "Mapping";                              // text - mapping
        // fs << "{" << "One" << 1;
        // fs <<        "Two" << 2 << "}";
        fs << "{";
        fs << "One" << 1;
        fs <<        "Two" << 2 ;
        fs<< "}";
        fs.release();                                       // explicit close
        cout << "Write Done." << endl;
    }
    {//read
        cout << endl << "Reading: " << endl;
        FileNode n;
        FileStorage fs;
        fs.open(filename, FileStorage::READ);
        n = fs["Mapping"];                                // Read mappings from a sequence
        cout << "Two  " << (int)(n["Two"]) << "; ";
        cout << "One  " << (int)(n["One"]) << endl << endl;
        fs.release();
    }
    */

    Json json(filename);
    string nodes[]={"dog","cat"};
    string childNodes[]={"x","y","w","h"};
    int contents[]={0,30,100,100,50,50,200,200};
    int readConts[8];

//    string nodes[]={"dog"};
//    string childNodes[]={"x","y","w","h"};
//    int contents[]={0,30,100,100};
//    int readConts[4];
    json.writeFile(nodes,len(nodes),childNodes,len(childNodes),contents,len(contents));
    json.readFile(nodes,len(nodes),childNodes,len(childNodes),readConts,len(readConts));

    pprint(readConts,len(readConts));
    cout << endl
        << "Tip: Open up " << filename << " with a text editor to see the serialized data." << endl;
    return 0;
}

