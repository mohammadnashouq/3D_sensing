//#include "MyApp.h"
#include <iostream>
#include <sstream>
#include <fstream>

#include <math.h>
#include <vector>

#include <array>
#include <list>
#include <tuple>
#include <imgui/imgui.h>

#include <Eigen/Dense>
#include "MyApp.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <sstream>
#include <fstream>
#include <string>
#include <locale>
#include <codecvt>
//#include <wx/utils.h> 
#include "Up_sampling.h"
#include "QualityMetrics.h"
#include "PointCloud3DConstraction.h"


using namespace std::chrono;

std::string WINAPI OpenFileDialog()
{

    std::wstringstream wss;
    std::wstring wstr; 


    HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED |
        COINIT_DISABLE_OLE1DDE);
    if (SUCCEEDED(hr))
    {
        IFileOpenDialog* pFileOpen;

        // Create the FileOpenDialog object.
        hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_ALL,
            IID_IFileOpenDialog, reinterpret_cast<void**>(&pFileOpen));

        if (SUCCEEDED(hr))
        {
            // Show the Open dialog box.
            hr = pFileOpen->Show(NULL);

            // Get the file name from the dialog box.
            if (SUCCEEDED(hr))
            {
                IShellItem* pItem;
                hr = pFileOpen->GetResult(&pItem);
                if (SUCCEEDED(hr))
                {
                    PWSTR pszFilePath;
                    hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszFilePath);

                    // Display the file name to the user.
                    if (SUCCEEDED(hr))
                    {
                        //MessageBoxW(NULL, pszFilePath, L"File Path", MB_OK);

                        wchar_t* localpszFilePath = pszFilePath;

                    

                        wss << localpszFilePath;

                        wstr = wss.str();
                        wss.str(L"");

                        CoTaskMemFree(pszFilePath);
                    }
                    pItem->Release();
                }
            }
            pFileOpen->Release();
        }
        CoUninitialize();
    }

    using convert_type = std::codecvt_utf8<wchar_t>;
    std::wstring_convert<convert_type, wchar_t> converter;

    //use converter (.to_bytes: wstr->str, .from_bytes: str->wstr)
    std::string converted_str = converter.to_bytes(wstr);


    return converted_str;
}






CMyApp::CMyApp(void)
{
  
    srand(time(0));
}


CMyApp::~CMyApp(void)
{
    std::cout << "dtor!\n";
}

bool CMyApp::Init()
{
    glClearColor(0.125f, 0.25f, 0.5f, 1.0f);

    glEnable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);

   

    return true;
}


void CMyApp::Clean()
{
    
}

double t = 0;

void CMyApp::Update()
{
    static Uint32 last_time = SDL_GetTicks();
    float delta_time = (SDL_GetTicks() - last_time) / 1000.0f;

   
}

void CMyApp::Render()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

   

   




    static int page = 0;
    bool open = true;
    ImGui::SetNextWindowPos(ImVec2(0, 0));
    ImGui::SetWindowSize(ImVec2(1024, 1024));
    if (ImGui::Begin("", &open, ImVec2(0, 0), 0.9f,   ImGuiWindowFlags_NoCollapse   | ImGuiWindowFlags_MenuBar)) {


        if (ImGui::CollapsingHeader("Filters"))
        {
            ImGui::Columns(2, "mixed");
            ImGui::Separator();





            ImGui::Text("Bilateral Filter ImageUrl");
            ImGui::PushItemWidth(500);
            if (ImGui::InputText("Url", Bilateral_Filter_Img, IM_ARRAYSIZE(Bilateral_Filter_Img))) {


                std::string myString(Bilateral_Filter_Img);
                bilateralFilter_input_img_url = myString;
            }

            ImGui::PushItemWidth(200);
            ImGui::SameLine();
            if (ImGui::Button("Browes Files"))
            {

                BrowseFileAndApplyDefaultFilter();


            }
            ImGui::Text("Set Segmas And Run Bilateral Filter  ");
            ImGui::Text("Spectral Segma  ");
            ImGui::PushItemWidth(100);
            ImGui::InputFloat("1 : ", &Spectral_Segma_1);
            ImGui::SameLine();
            ImGui::InputFloat("2 : ", &Spectral_Segma_2);
            ImGui::InputFloat("3 : ", &Spectral_Segma_3);

            ImGui::SameLine();
            ImGui::InputFloat("4 : ", &Spectral_Segma_4);

            ImGui::Text("Special Segma  ");
            ImGui::InputFloat("1 : ", &Special_Segma_1);
            ImGui::SameLine();
            ImGui::InputFloat("2 : ", &Special_Segma_2);
            ImGui::InputFloat("3 : ", &Special_Segma_3);
            ImGui::SameLine();
            ImGui::InputFloat("4 : ", &Special_Segma_4);

            if (ImGui::Button("Run Bilateral Filter"))
            {
                CMyApp::RunBilateralFilter();
            }


            ImGui::NextColumn();

            ImGui::PushItemWidth(200);

             


        

            ImGui::Text("Hello");
            GLuint BFtexture;
            glGenTextures(1, &BFtexture);
            glBindTexture(GL_TEXTURE_2D, BFtexture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, noiseImg.cols, noiseImg.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, noiseImg.data);
            ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(BFtexture)), ImVec2(noiseImg.cols / 3, noiseImg.rows / 3));

            if (ImGui::InputInt("Mean", &noiseMean, 1, 1, 0))
            {
                CMyApp::ApplyPreFilterImgNoise();
            }

            if (ImGui::InputInt("STD", &noiseStddev, 1, 1, 0)) {
                CMyApp::ApplyPreFilterImgNoise();
            }

            ImGui::NextColumn();


        }
        if (ImGui::CollapsingHeader("UpSampling")) 
        {
            ImGui::Columns(2, "mixed");
            ImGui::Separator();





            ImGui::Text("Low Disparity Image");
            ImGui::PushItemWidth(500);
            ImGui::InputText("Url", _Low_Disparty_Img_Url, IM_ARRAYSIZE(_Low_Disparty_Img_Url));

            ImGui::PushItemWidth(200);
            ImGui::SameLine();
            if (ImGui::Button("Browes Files"))
            {
                std::string filePath = OpenFileDialog();
                strcpy(_Low_Disparty_Img_Url, filePath.c_str());
                Input_Low_DisparityImage = cv::imread(_Low_Disparty_Img_Url, cv::IMREAD_COLOR);
                cv::cvtColor(Input_Low_DisparityImage, Input_Low_DisparityImage, cv::COLOR_BGR2RGBA);
            }
           
          
            GLuint Low_DisparityImagetexture;
            glGenTextures(1, &Low_DisparityImagetexture);
            glBindTexture(GL_TEXTURE_2D, Low_DisparityImagetexture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, Input_Low_DisparityImage.cols, Input_Low_DisparityImage.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, Input_Low_DisparityImage.data);
            ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(Low_DisparityImagetexture)), ImVec2(Input_Low_DisparityImage.cols / 3, Input_Low_DisparityImage.rows / 3));








            ImGui::Text("RGB Image");
            ImGui::PushItemWidth(500);
            ImGui::InputText("Url", _RGB_Img_Url, IM_ARRAYSIZE(_RGB_Img_Url));
            ImGui::PushItemWidth(200);
            ImGui::SameLine();
            if (ImGui::Button("Browes Files2"))
            {
                std::string filePath = OpenFileDialog();
                strcpy(_RGB_Img_Url, filePath.c_str());

                InputRGBImage = cv::imread(_RGB_Img_Url, cv::IMREAD_COLOR);
                cv::cvtColor(InputRGBImage, InputRGBImage, cv::COLOR_BGR2RGBA);
            }
           
            
            GLuint RGBImagetexture;
            glGenTextures(1, &RGBImagetexture);
            glBindTexture(GL_TEXTURE_2D, RGBImagetexture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, InputRGBImage.cols, InputRGBImage.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, InputRGBImage.data);
            ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(RGBImagetexture)), ImVec2(InputRGBImage.cols / 3, InputRGBImage.rows / 3));





            ImGui::Text("Heigh resultion Diparity Image");
            ImGui::PushItemWidth(500);
            ImGui::InputText("Url", _Heigh_Disparty_Img_Url, IM_ARRAYSIZE(_Heigh_Disparty_Img_Url));
            ImGui::PushItemWidth(200);
            ImGui::SameLine();
            if (ImGui::Button("Browes Files3"))
            {
                std::string filePath = OpenFileDialog();
                strcpy(_Heigh_Disparty_Img_Url, filePath.c_str());
                Input_Heigh_Disparty_Img = cv::imread(_Heigh_Disparty_Img_Url, cv::IMREAD_COLOR);
            }
          
            if (!Input_Heigh_Disparty_Img.empty()) {
                cv::cvtColor(Input_Heigh_Disparty_Img, Input_Heigh_Disparty_Img, cv::COLOR_BGR2RGBA);
            }
            GLuint Input_Heigh_Dispartytexture;
            glGenTextures(1, &Input_Heigh_Dispartytexture);
            glBindTexture(GL_TEXTURE_2D, Input_Heigh_Dispartytexture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, Input_Heigh_Disparty_Img.cols, Input_Heigh_Disparty_Img.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, Input_Heigh_Disparty_Img.data);
            ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(Input_Heigh_Dispartytexture)), ImVec2(Input_Heigh_Disparty_Img.cols / 3, Input_Heigh_Disparty_Img.rows / 3));





            ImGui::Text("Output Folder Name : ");
            ImGui::PushItemWidth(200);
            ImGui::InputText("Name", _OutputFolder_Name, IM_ARRAYSIZE(_OutputFolder_Name));



            ImGui::Text("Dmin : ");
            ImGui::InputInt("Dmin", &Dmin, 1, 1, 0);

            if (ImGui::Button("Run UpSampling and 3D Constraction"))
            {
                CMyApp::RunUpSampling_AndPointCloud3dConstraction();
            }


            ImGui::Text("Run UpSampling and 3D Constraction for entier dataset : ");

            if (ImGui::Button("Run UpSampling and 3D Constraction for entier dataset"))
            {
                CMyApp::RunUpSampling_AndPointCloud3dConstraction_full_dataset();
            }

            ImGui::NextColumn();


            ImGui::Text("Iterative UpSampled  Image");
          
            cv::Mat Iterative_UpSampled_Disparty_Img_cloned;
            if (!Iterative_UpSampled_Disparty_Img.empty()) {
                Iterative_UpSampled_Disparty_Img_cloned = Iterative_UpSampled_Disparty_Img.clone();
                cv::cvtColor(Iterative_UpSampled_Disparty_Img_cloned, Iterative_UpSampled_Disparty_Img_cloned, cv::COLOR_BGR2RGBA);
            }
            GLuint Iterative_UpSampledtexture;
            glGenTextures(1, &Iterative_UpSampledtexture);
            glBindTexture(GL_TEXTURE_2D, Iterative_UpSampledtexture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, Iterative_UpSampled_Disparty_Img_cloned.cols, Iterative_UpSampled_Disparty_Img_cloned.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, Iterative_UpSampled_Disparty_Img_cloned.data);
            ImGui::Image(reinterpret_cast<void*>(static_cast<intptr_t>(Iterative_UpSampledtexture)), ImVec2(Iterative_UpSampled_Disparty_Img_cloned.cols / 3, Iterative_UpSampled_Disparty_Img_cloned.rows / 3));


            ImGui::Text("Quality Metrcies Between JB UpSampled Disparity and Ground Truth : ");
            ImGui::Text(Quality_Metrcies_JBUpSampled_ground);

            ImGui::Text("Quality Metrcies Between Iterative UpSampled Disparity and Ground Truth : ");
            ImGui::Text(Quality_Metrcies_IterativeUpSampled_ground);



            




            // the resaulted upsampled image 
        }
        if (ImGui::CollapsingHeader("Test Window")) {
            ImGui::ShowTestWindow();
        }
   }

  
    ImGui::End();

}

void CMyApp::BrowseFileAndApplyDefaultFilter() {
    std::string filePath = OpenFileDialog();

    if (filePath.empty()) {
        return;
    }

    strcpy(Bilateral_Filter_Img, filePath.c_str());


    bilateralFilter_input_img_url = filePath;

    Filter_input_MatIMG = cv::imread(bilateralFilter_input_img_url, cv::IMREAD_COLOR);

    noiseImg = Filter_input_MatIMG.clone();

    if (Filter_input_MatIMG.data == nullptr) {
        std::cerr << "Failed to load image" << std::endl;
        return;
    }

    //cv::imshow("im", im);
    //cv::waitKey();

    cv::Mat noise(Filter_input_MatIMG.size(), Filter_input_MatIMG.type());
    uchar mean = 0;
    uchar stddev = 25;
    cv::randn(noise, mean, stddev);


    noiseImg += noise;

    cv::cvtColor(noiseImg, noiseImg, cv::COLOR_BGR2RGBA);

}

void CMyApp::ApplyPreFilterImgNoise() {
    noiseImg = Filter_input_MatIMG.clone();

    if (Filter_input_MatIMG.data == nullptr) {
        std::cerr << "Failed to load image" << std::endl;
    }

    //cv::imshow("im", im);
    //cv::waitKey();

    cv::Mat noise(Filter_input_MatIMG.size(), Filter_input_MatIMG.type());
    uchar mean = 0;
    uchar stddev = 25;
    cv::randn(noise, mean, stddev);


    noiseImg += noise;

    cv::cvtColor(noiseImg, noiseImg, cv::COLOR_BGR2RGBA);
}


void CMyApp::RunBilateralFilter()
{
   // bool Generate_OpenCV_Disparity = false;

    if (bilateralFilter_input_img_url.empty()) {
        MessageBox(NULL, "Please Enter Image Url", "Error", MB_ICONINFORMATION);
        return;
      
    }

    Filter_input_MatIMG = cv::imread(bilateralFilter_input_img_url, 0);

    if (Filter_input_MatIMG.data == nullptr) {
        std::cerr << "Failed to load image" << std::endl;

        MessageBox(NULL, "Failed to load image", "Error", MB_ICONINFORMATION);
        return;
    }

    //cv::imshow("im", im);
    //cv::waitKey();

    cv::Mat noise(Filter_input_MatIMG.size(), Filter_input_MatIMG.type());

    cv::randn(noise, noiseMean, noiseStddev);


    Filter_input_MatIMG += noise;

    //cv::imshow("im", im);
    //cv::waitKey();
    int height = Filter_input_MatIMG.size().height;
    int width = Filter_input_MatIMG.size().width;

    cv::Mat output = cv::Mat::zeros(height, width, CV_8UC1);

    std::string disparity_window = "disparity";




    // bilateral
    /*double window_size = 11;
    cv::bilateralFilter(im, output, window_size, 2 * window_size, window_size / 2);
    cv::imshow("bilateral", output);*/

    float Spectral_Segmas[4] = { Spectral_Segma_1 , Spectral_Segma_2 ,Spectral_Segma_3 , Spectral_Segma_4 };
    float Special_Segmas[4] = { Special_Segma_1 , Special_Segma_2 ,Special_Segma_3 , Special_Segma_4 };

    std::stringstream string_stream;

    string_stream << "log.txt";
    std::string logUrl = string_stream.str();
    string_stream.str("");

    std::ofstream logFile;
    logFile.open(logUrl);

    std::cout << "-------------------- bilateralFilter ----------------------------" << std::endl;
    logFile << "-------------------- bilateralFilter ----------------------------" << std::endl;

    int arr_size = sizeof(Spectral_Segmas) / sizeof(float);


    double MinMetSum = DBL_MAX;
    float Best_SegmaSpectral = 0;
    float Best_SegmaSpecial = 0;


    for (int i_spectral = 0; i_spectral < arr_size; i_spectral++) {
        for (int j_Special = 0; j_Special < arr_size; j_Special++) {



            int num = arr_size * i_spectral + j_Special;



            cv::Mat output = cv::Mat::zeros(height, width, CV_8UC1);

            float SegmaSpectral = Spectral_Segmas[i_spectral];
            float SegmaSpecial = Special_Segmas[j_Special];
            std::cout << "----------------------------------------------------------------------------------" << std::endl;
            logFile << "----------------------------------------------------------------------------------" << std::endl;

            std::cout << "num : " << num << "runing bilateral Filter Segma Spectral =  " << SegmaSpectral << "and Special Segmas = " << SegmaSpecial << std::endl;
            logFile << "num : " << num << "runing bilateral Filter Segma Spectral =  " << SegmaSpectral << "and Special Segmas = " << SegmaSpecial << std::endl;



            string_stream << "Bilateral Filter Spectral = " << SegmaSpectral << " Special : " << SegmaSpecial;
            std::string imShowWindowName   = string_stream.str();
            string_stream.str("");

            OurFilter_Bilateral(Filter_input_MatIMG, output, SegmaSpectral, SegmaSpecial);
           //  cv::imshow(imShowWindowName, output);







            string_stream << "output/" << "num_" << num << "Spectral_" << SegmaSpectral << "_Special_" << SegmaSpecial << "_img.png";
            std::string imageUrl = string_stream.str();

            cv::imwrite(imageUrl, output);


            std::cout << "--bilateral Filter Quailty Metrices between the Image and the Fillterd Image---" << std::endl;
            logFile << "--bilateral Filter Quailty Metrices between the Image and the Fillterd Image---" << std::endl;

            double MetrciesSum = 0;
            CMyApp::QuailtyMetrices(Filter_input_MatIMG, output, logFile, MetrciesSum);
            if (MetrciesSum < MinMetSum) {
                MinMetSum = MetrciesSum;
                Best_SegmaSpecial = SegmaSpecial;
                Best_SegmaSpectral = SegmaSpectral;
            }



            string_stream.str("");

          

        }
    }


    std::cout << "--bilateral Filter Best Segmas Values Spectral Segma : " << Best_SegmaSpectral<<" , Best Special Segma : "<< Best_SegmaSpecial << std::endl;
    logFile << "--bilateral Filter Best Segmas Values Spectral Segma : " << Best_SegmaSpectral << " , Best Special Segma : " << Best_SegmaSpecial << std::endl;
    system("explorer.exe output");




    logFile.close();
}


void CMyApp::RunUpSampling_AndPointCloud3dConstraction()
{

    std::stringstream string_stream;
    std::string Low_Disparty_Img_Url(_Low_Disparty_Img_Url);
    std::string Heigh_Disparty_Img_Url(_Heigh_Disparty_Img_Url);
    std::string RGB_Img_Url(_RGB_Img_Url);

    cv::Mat In_Low_DisparityImage = cv::imread(Low_Disparty_Img_Url, 0);
    cv::Mat In_Heigh_Disparty_Img = cv::imread(Heigh_Disparty_Img_Url, 0);
    cv::Mat InRGBImage = cv::imread(RGB_Img_Url, 0);

     if (In_Low_DisparityImage.empty()) {
         MessageBox(NULL, "Faild to Load Low Disparity Image !", "Error", MB_ICONINFORMATION);
         return;
     }

     if (In_Heigh_Disparty_Img.empty()) {
         MessageBox(NULL, "Faild to Load Heigh Disparity Image !", "Error", MB_ICONINFORMATION);
         return;
     }

     if (InRGBImage.empty()) {
         MessageBox(NULL, "Faild to Load RGB Image !", "Error", MB_ICONINFORMATION);
         return;
     }
    


     string_stream << "output_stereo2/" << _OutputFolder_Name <<"/"<< "_log.txt";
     std::string logUrl = string_stream.str();
     string_stream.str("");

     std::ofstream logFile;
     logFile.open(logUrl);

     std::cout << "Starting Disparty Upsampling and 3D Constraction "<<std::endl;
     logFile << "Starting Disparty Upsampling and 3D Constraction " << std::endl;

     std::cout << "Disparity Input :  " << Low_Disparty_Img_Url<< std::endl;
     logFile << "Disparity Input :  " << Low_Disparty_Img_Url << std::endl;

     std::cout << "RGB Image Input :  " << RGB_Img_Url << std::endl;
     logFile << "RGB Image Input :  " << RGB_Img_Url << std::endl;

     std::chrono::steady_clock::time_point start;
     std::chrono::steady_clock::time_point stop;
     std::chrono::microseconds duration;

     // Starting JB Upsampling for the Ground Truth Image

     start = high_resolution_clock::now();

    JB_UpSampled_Disparty_Img =  Joint_BilateralUpsampling(InRGBImage , In_Low_DisparityImage);

    stop =  high_resolution_clock::now();

    duration = duration_cast<microseconds>(stop - start);

    std::cout << "JB UpSampling execution time : " << duration.count() << std::endl;
    logFile << "JB UpSampling execution time : " << duration.count() << std::endl;



    string_stream << "output_stereo2/" << _OutputFolder_Name <<"/"<< "JB_UpSampled_Disparity.png";
    std::string JB_UpSampled_DisparityUrl = string_stream.str();
    string_stream.str("");

    std::cout << "JB UpSampling Quailty Metrices between the Upsampled Disparity Image and the Ground Truth" << std::endl;
    logFile << "JB UpSampling Quailty Metrices between the Upsampled Disparity Image and the Ground Truth" << std::endl;
    std::cout << "----------------------------------------------------------------------------" << std::endl;
    logFile << "------------------------------------------------------------------------------" << std::endl;
    double MetTotal = 0;
    std::string  JBQuailtyMetricesStr = CMyApp::QuailtyMetrices(In_Heigh_Disparty_Img, JB_UpSampled_Disparty_Img,logFile, MetTotal);
    strcpy(Quality_Metrcies_JBUpSampled_ground, JBQuailtyMetricesStr.c_str());


    cv::imwrite(JB_UpSampled_DisparityUrl, JB_UpSampled_Disparty_Img);

    std::cout << "----------------------------------------------------------------------------" << std::endl;
    logFile << "------------------------------------------------------------------------------" << std::endl;


    // Starting Iterative Upsampling for the Ground Truth Image

    start = high_resolution_clock::now();
    Iterative_UpSampled_Disparty_Img = IterativeUpsampling(InRGBImage, In_Low_DisparityImage);

    stop = high_resolution_clock::now();

    duration = duration_cast<microseconds>(stop - start);

    std::cout << "Iterative UpSampling execution time : " << duration.count() << std::endl;
    logFile << "Iterative UpSampling execution time : " << duration.count() << std::endl;



    string_stream << "output_stereo2/" << _OutputFolder_Name <<"/"<< "Iterative_UpSampled_Disparity.png";
    std::string Iterative_UpSampled_DisparityUrl = string_stream.str();
    string_stream.str("");


    std::cout << "Iterative UpSampling Quailty Metrices between the Upsampled Disparity Image and the Ground Truth" << std::endl;
    logFile << "Iterative UpSampling Quailty Metrices between the Upsampled Disparity Image and the Ground Truth" << std::endl;
    std::cout << "----------------------------------------------------------------------------" << std::endl;
    logFile << "------------------------------------------------------------------------------" << std::endl;
   
   std::string  IterativeQuailtyMetricesStr =  CMyApp::QuailtyMetrices(In_Heigh_Disparty_Img, Iterative_UpSampled_Disparty_Img, logFile, MetTotal);
   strcpy(Quality_Metrcies_IterativeUpSampled_ground, IterativeQuailtyMetricesStr.c_str());

    std::cout << "----------------------------------------------------------------------------" << std::endl;
    logFile << "------------------------------------------------------------------------------" << std::endl;

    cv::imwrite(Iterative_UpSampled_DisparityUrl, Iterative_UpSampled_Disparty_Img);




    // Starting 3D Point Cloud Constraction from the UpSampled Disparity Image Generated by the Iterative approach 



   // Disparity2PointCloud

    string_stream << "output_stereo2/" << _OutputFolder_Name <<"/"<< "PointCloud3D.xyz";
    std::string PointCloud3D_outputFile_url = string_stream.str();
    string_stream.str("");

    string_stream << "output_stereo2/" << _OutputFolder_Name << "/" << "PointCloud3D_W_Norm.ply";
    std::string PointCloud3D_outputFile_W_Norm_url = string_stream.str();
    string_stream.str("");

    Disparity2PointCloud(PointCloud3D_outputFile_url, PointCloud3D_outputFile_W_Norm_url, Iterative_UpSampled_Disparty_Img.size().height, Iterative_UpSampled_Disparty_Img.size().width, Iterative_UpSampled_Disparty_Img, 3, Dmin, baseline, focal_length,logFile);

    std::cout << " Disparty Upsampling and 3D Constraction completed ------------- " << std::endl;
    logFile << " Disparty Upsampling and 3D Constraction completed -------------- " << std::endl;

    logFile.flush();
    logFile.close();


   /* string_stream << "explorer.exe output_stereo2/" << _OutputFolder_Name;
    std::string OutputFolder_cmd = string_stream.str();
    string_stream.str("");

    system(OutputFolder_cmd.c_str());*/
   
}


void CMyApp::RunUpSampling_AndPointCloud3dConstraction_full_dataset() 
{
    for (int vn = 1; vn < 5; vn++) {


        int input_num = vn;





        std::stringstream string_stream;

        string_stream << "input_stereo2/" << input_num << "/view0.png";
        std::string image1Url = string_stream.str();
        string_stream.str("");

        string_stream << "input_stereo2/" << input_num << "/view1.png";
        std::string image2Url = string_stream.str();
        string_stream.str("");

        string_stream << "input_stereo2/" << input_num << "/ground.png";
        std::string groundTruthUrl = string_stream.str();
        string_stream.str("");

       

        string_stream << "input_stereo2/" << input_num << "/low_disp_ground.png";
        std::string in_low_disparity_groundUrl = string_stream.str();
        string_stream.str("");

        string_stream << "input_stereo2/" << input_num << "/dmin.txt";
        std::string dminUrl = string_stream.str();
        string_stream.str("");

        string_stream << input_num;
        std::string outputFolderStr = string_stream.str();
        string_stream.str("");

        std::ifstream dminfile(dminUrl);
        std::string dminstr;
        std::getline(dminfile, dminstr);

        int dmin = stoi(dminstr);

        dminfile.close();

        strcpy(_Low_Disparty_Img_Url, in_low_disparity_groundUrl.c_str());
        strcpy(_Heigh_Disparty_Img_Url, groundTruthUrl.c_str());
        strcpy(_RGB_Img_Url, image1Url.c_str());
        strcpy(_OutputFolder_Name, outputFolderStr.c_str());

        
        Dmin = dmin;



        CMyApp::RunUpSampling_AndPointCloud3dConstraction();

    }
}


std::string CMyApp::QuailtyMetrices(cv::Mat& Img1, cv::Mat& Img2 , std::ofstream& logFile,double& Total_MetrciesSum)
{
    double _SSD = SSD(Img1, Img2);
    double _MSE = MSE(Img1, Img2);
    double _RMSE = RMSE(Img1, Img2);
    double _PSNR = PSNR(Img1, Img2);
  
    double _SSIM = SSIM(Img1, Img2);


    Total_MetrciesSum = _SSD + _MSE + _RMSE + _PSNR + _SSIM;

    std::stringstream ss;

    std::cout << "SSD : " << _SSD << std::endl;
    ss << "SSD : " << _SSD << std::endl;

    std::cout << "MSE : " << _MSE << std::endl;
    ss << "MSE : " << _MSE << std::endl;

    std::cout << "RMSE : " << _RMSE << std::endl;
    ss << "RMSE : " << _RMSE << std::endl;

    std::cout << "PSNR : " << _PSNR << std::endl;
    ss << "PSNR : " << _PSNR << std::endl;

    std::cout << "SSIM : " << _SSIM << std::endl;
    ss << "SSIM : " << _SSIM << std::endl;

    logFile << ss.str();

    return ss.str();
}



void CMyApp::KeyboardDown(SDL_KeyboardEvent& key)
{
    
}

void CMyApp::KeyboardUp(SDL_KeyboardEvent& key)
{
  
}

void CMyApp::MouseMove(SDL_MouseMotionEvent& mouse)
{
  
}

void CMyApp::MouseDown(SDL_MouseButtonEvent& mouse)
{
}

void CMyApp::MouseUp(SDL_MouseButtonEvent& mouse)
{
}

void CMyApp::MouseWheel(SDL_MouseWheelEvent& wheel)
{
}

void CMyApp::Resize(int _w, int _h)
{
    glViewport(0, 0, _w, _h);

   
}