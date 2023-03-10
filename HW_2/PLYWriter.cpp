/* PLY format writer
 *
 * Implemented by Levente Hajder
 * hajder@inf.elte.hu
 * 30-06-2021
 */

#include "PLYWriter.h"


 /* Write a PLY file
  *
  * Input:
  *
  * fileName: nam of the file written
  * vector<Point3f> pts: spatial points
  * vector<Point3i> colors: colors. RGB color values are given by coordinates of Point3i
  *
  *
  */



void WritePLY(const char* fileName, vector<Point3f> pts, vector<Point3f> Norms) {
    int num = pts.size();
    ofstream myfile;
    myfile.open(fileName);

    myfile << "ply\n";
    myfile << "format ascii 1.0\n";
    myfile << "element vertex " << num << endl;
    myfile << "property float x\n";
    myfile << "property float y\n";
    myfile << "property float z\n";
    myfile << "property float nx\n";
    myfile << "property float ny\n";
    myfile << "property float nz\n";
    myfile << "end_header\n";


    for (int idx = 0; idx < num; idx++) {
        Point3f point = pts.at(idx);
        Point3f norm = Norms.at(idx);

        myfile << point.x << " " << point.y << " " << point.z << " " << norm.x << " " << norm.y << " " << norm.z << endl;

    }

    myfile.close();



}
