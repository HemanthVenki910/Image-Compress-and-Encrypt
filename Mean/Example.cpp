#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <ctime>
#include <vector>

using namespace cv;

Mat Apply_Mean_Reduce(Mat Image,int size)
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
			MatrixXd Mat;

			cv::cv2eigen(temp, Mat);
			int i,j;
			int Mean=0;
			int Minimum=1000;

			for(i=0;i<size;i++)
				for(j=0;j<size;j++)
				{
					Mean+=Mat(i,j);
					if(Minimum>Mat(i,j))
						Minimum=Mat(i,j);
				}
			
			Mean=Mean/(size*size);

			for(i=0;i<size;i++)
				for(j=0;j<size;j++)
				{
					if(Mean>100)
						continue;
					if(Mat(i,j)<Mean)
						Mat(i,j)=Minimum;
					if(Mean<50)
						Mat(i,j)=0;
				}
			cv::eigen2cv(Mat, temp);
			temp=temp.reshape(chs,0);
			RGBChannels[n]=temp;
		}

		std::vector <Mat> channels;
		channels.push_back(RGBChannels[0]);
		channels.push_back(RGBChannels[1]);
		channels.push_back(RGBChannels[2]);
		cv::merge(channels,Image);

	}
	return Image;
}
Mat Mean_Reduction(Mat Image,int size)
{
	int rows=Image.rows;
	int cols=Image.cols;
	int i;
	int j;
	int row_end=0;
	int col_end=0;

	Mat Resultant_Image=Image.clone();
	Mat Reduced_Image;

	for(;row_end<rows;)
	{
		i=row_end;
		row_end+=size;
		if(row_end>rows)
			break;
		col_end=0;
		for(;col_end<cols;)
		{

			j=col_end;
			col_end+=size;
			if(col_end>cols)
				break;
			Reduced_Image=Image(Range(i,row_end),Range(j,col_end));		
			Reduced_Image=Apply_Mean_Reduce(Reduced_Image,size);			
			Reduced_Image.copyTo(Resultant_Image.rowRange(i,row_end).colRange(j, col_end));			
		}
		continue;	
	}
	return Resultant_Image;
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
	imwrite("Actual_Image.jpg",Image);
	Image=Mean_Reduction(Image,10);
	imwrite("Reduced_Image.jpg",Image);
	cv::waitKey(0);
	return 0;
}

