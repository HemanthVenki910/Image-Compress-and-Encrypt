#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <ctime>
#include <vector>

using namespace cv;

Mat Ceaser_EnCode(Mat Image,int *key)
{
	{
		std::vector <Mat> RGBChannels(3);
		cv::Mat temp;
		cv::split(Image,RGBChannels);
		for(int n=0;n<3;n++)
		{
			temp=RGBChannels[n];
			int chs = temp.channels();
			cv::Mat Img = temp.reshape(1, 0);
			typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
			MatrixXd mat;

			cv::cv2eigen(temp, mat);
			int r=temp.rows;
			int c=temp.cols;
			int ad;
			for(int j=0;j<r;j++)
				for(int k=0;k<c;k++)
				{
					ad=mat(j,k);
					mat(j,k)=(ad+key[n])%256;
				}
			cv::eigen2cv(mat, temp);
			temp=temp.reshape(chs,0);
			RGBChannels[n]=temp;
		}
		std::vector <Mat> channels;
		channels.push_back(RGBChannels[0]);
		channels.push_back(RGBChannels[1]);
		channels.push_back(RGBChannels[2]);
		cv::merge(channels,Image);
	}
	cv::imshow("Red",Image*(1.0/255));
	waitKey(0);
	return Image;
}
Mat Decrypt(Mat Image,int *key)
{
	{
		std::vector <Mat> RGBChannels(3);
		cv::Mat temp;
		cv::split(Image,RGBChannels);
		for(int n=0;n<3;n++)
		{
			temp=RGBChannels[n];
			int chs = temp.channels();
			cv::Mat Img = temp.reshape(1, 0);
			typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;
			MatrixXd mat;

			cv::cv2eigen(temp, mat);
			int r=temp.rows;
			int c=temp.cols;
			int org;
			for(int j=0;j<r;j++)
				for(int k=0;k<c;k++)
				{
					org=mat(j,k)-*(key+n);
					if(org<0)
						mat(j,k)=mat(j,k)+256-*(key+n);
					else
						mat(j,k)=org;
				}
			cv::eigen2cv(mat, temp);
			temp=temp.reshape(chs,0);
			RGBChannels[n]=temp;

		}
		std::vector <Mat> channels;
		channels.push_back(RGBChannels[0]);
		channels.push_back(RGBChannels[1]);
		channels.push_back(RGBChannels[2]);
		cv::merge(channels,Image);
	}
	cv::imshow("Red",Image*(1.0/255));
	waitKey(0);
	return Image;
}
int main(int argc, char** argv )
{
	srand(time(0));
	if ( argc != 2 )
	{
		printf("Please Mention the Image to be Read \n");  
		return -1;
	}
	Mat Image;
	Image = imread(argv[1], 1);
	if ( !Image.data )
	{
		printf("No image data \n");
		return -1;
	}

	int key[3];
	for(int i=0;i<3;i++)
		key[i]=rand()%256;

	Image=Ceaser_EnCode(Image,key);
	Image=Decrypt(Image,key);
	return 0;
}

