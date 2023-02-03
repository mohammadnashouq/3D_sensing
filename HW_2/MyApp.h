#pragma once

// C++ includes
#include <memory>

// GLEW
#include <GL/glew.h>

// SDL
#include <SDL.h>
#include <SDL_opengl.h>


#include <windows.h>
#include <shobjidl.h> 
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>



std::string OpenFileDialog();




class CMyApp
{
public:

	std::string bilateralFilter_input_img_url;

	cv::Mat Filter_input_MatIMG;

	cv::Mat InputRGBImage;

	cv::Mat Input_Low_DisparityImage;
	cv::Mat Input_Heigh_Disparty_Img;

	 char Bilateral_Filter_Img[512] = "Bilateral Filter ImageUrl";



	 char _Low_Disparty_Img_Url[1024] = "Low Disparty Image";
	 char _Heigh_Disparty_Img_Url[1024] = "Heigh Disparty Image";
	 char _RGB_Img_Url[1024] = "Rgb Image Url";
	 char _OutputFolder_Name[1024] = "_OutputFolder_Name";




	 char Quality_Metrcies_JBUpSampled_ground[10000] = "";
	 char Quality_Metrcies_IterativeUpSampled_ground[10000] = "";

	 cv::Mat JB_UpSampled_Disparty_Img;
	 cv::Mat Iterative_UpSampled_Disparty_Img;




	cv::Mat noiseImg;

	int noiseMean = 0;
	int noiseStddev = 25;

	int Dmin = 200;


	const double focal_length = 3740;
	const double baseline = 160;


	float Spectral_Segma_1 = 5;
	float Spectral_Segma_2 = 9;
	float Spectral_Segma_3 = 10;
	float Spectral_Segma_4 = 12;

	float Special_Segma_1 = 5;
	float Special_Segma_2 = 9;
	float Special_Segma_3 = 10;
	float Special_Segma_4 = 12;






	



	

	CMyApp(void);
	~CMyApp(void);


	void ApplyPreFilterImgNoise();


	void RunBilateralFilter();


	void BrowseFileAndApplyDefaultFilter();


	void RunUpSampling_AndPointCloud3dConstraction();

	void RunUpSampling_AndPointCloud3dConstraction_full_dataset();


	std::string QuailtyMetrices(cv::Mat& Img1 , cv::Mat& Img2 , std::ofstream& logFile , double& Total_MetrciesSum);
	

	bool Init();
	void Clean();

	void Update();
	void Render();

	void KeyboardDown(SDL_KeyboardEvent&);
	void KeyboardUp(SDL_KeyboardEvent&);
	void MouseMove(SDL_MouseMotionEvent&);
	void MouseDown(SDL_MouseButtonEvent&);
	void MouseUp(SDL_MouseButtonEvent&);
	void MouseWheel(SDL_MouseWheelEvent&);
	void Resize(int, int);
protected:
	// helper methods
	void TextureFromFileAttach(const char* filename, GLuint role) const;

	// shader variables


	// raw OpenGL stuff
	GLuint				m_skyboxTexture;
};

