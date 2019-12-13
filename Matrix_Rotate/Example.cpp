#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <ctime>
#include <vector>
#include <cmath>
using namespace cv;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;

MatrixXd Rotate_Individual(MatrixXd Matrix,int size,int Key)
{
	int i=0,j=0;
	int temp;

	temp=Key;
	int Binary_key[size];
	
	while(temp!=0)
	{
		Binary_key[i++]=temp%2;
		temp=temp/2;
	}	
	for(j=0;j<size/2;j++)
	{
		if(Binary_key[j]==1)
		{
			for(i=j;i<size-j-1;i++)
			{
				temp=Matrix(j,i);
				Matrix(j,i)=Matrix(size-i-1,j);
				Matrix(size-i-1,j)=Matrix(size-j-1,size-i-1);
				Matrix(size-j-1,size-i-1)=Matrix(i,size-j-1);
				Matrix(i,size-j-1)=temp;	
			}
		}
		else
		{
			int tryx;
			for(tryx=0;tryx<3;tryx++)
			{
				for(i=j;i<size-j-1;i++)
				{
					temp=Matrix(j,i);
					Matrix(j,i)=Matrix(size-i-1,j);
					Matrix(size-i-1,j)=Matrix(size-j-1,size-i-1);
					Matrix(size-j-1,size-i-1)=Matrix(i,size-j-1);
					Matrix(i,size-j-1)=temp;	
				}
			}
			
		}
	}
/*
	int ad;
	int temp_Key=Key%256;
	for(i=0;i<size;i++)
		for(j=0;j<size;j++)
		{
			ad=Matrix(i,j);
			Matrix(i,j)=(ad+temp_Key)%256;
		}
*/
	return Matrix;
}

Mat Rotate_Sub_Matrix(Mat Image,int size,std::vector<int>& keys)
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
			MatrixXd Mat;

			cv::cv2eigen(temp, Mat);
			
			int divi=(pow(2,size-1));
			keys[n]=(rand()%divi);
			Mat=Rotate_Individual(Mat,size,keys[n]);

			cv::eigen2cv(Mat, temp);
			temp=temp.reshape(chs,0);
			RGBChannels[n]=temp;
		}
		std::vector <Mat> channels;
		channels.push_back(RGBChannels[0]);
		channels.push_back(RGBChannels[1]);
		channels.push_back(RGBChannels[2]);
		cv::merge(channels,Image);
		std::cout<<keys[0]<<" "<<keys[1]<<" "<<keys[2]<<std::endl;
	}
	return Image;
}

Mat Matrix_Rotate(Mat Image,int size,std::vector<std::vector <int> >& Keys)
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
			std::vector<int> sub_keys={0,0,0};
			Reduced_Image=Image(Range(i,row_end),Range(j,col_end));		
			Reduced_Image=Rotate_Sub_Matrix(Reduced_Image,size,sub_keys);			
			Keys.push_back(sub_keys);			
			Reduced_Image.copyTo(Resultant_Image.rowRange(i,row_end).colRange(j, col_end));			
		}
		continue;	
	}
	return Resultant_Image;
}

//Getting back the actual Image back from the Encrypted Image
MatrixXd Rotate_Individual_Back(MatrixXd Matrix,int size,int Key)
{
	int i=0,j=0;
	int temp;

	temp=Key;
	int Binary_key[size]={0};
	
	while(temp!=0)
	{
		Binary_key[i++]=temp%2;
		temp=temp/2;
	}
	MatrixXd Temp;	
	for(j=0;j<size/2;j++)
	{
		if(Binary_key[j]==0)
		{
			for(i=j;i<size-j-1;i++)
			{
				temp=Matrix(j,i);
				Matrix(j,i)=Matrix(size-i-1,j);
				Matrix(size-i-1,j)=Matrix(size-j-1,size-i-1);
				Matrix(size-j-1,size-i-1)=Matrix(i,size-j-1);
				Matrix(i,size-j-1)=temp;	
			}

		}
		else
		{
			int tryx;
				for(tryx=0;tryx<3;tryx++)
				{
					for(i=j;i<size-j-1;i++)
					{
						temp=Matrix(j,i);
						Matrix(j,i)=Matrix(size-i-1,j);
						Matrix(size-i-1,j)=Matrix(size-j-1,size-i-1);
						Matrix(size-j-1,size-i-1)=Matrix(i,size-j-1);
						Matrix(i,size-j-1)=temp;	
					}
				}			
		}
	}
/*
	int org;
	int temp_Key=Key%256;
	for(i=0;i<size;i++)
		for(j=0;j<size;j++)
		{
			org=Matrix(i,j)-temp_Key;
			if(org<0)
				Matrix(i,j)=Matrix(i,j)+256-temp_Key;
			else
				Matrix(i,j)=org;
		}
*/
	return Matrix;
}
Mat Rotate_Sub_Matrix_Back(Mat Image,int size,std::vector<int>& keys)
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
			MatrixXd Mat;

			cv::cv2eigen(temp, Mat);
			Mat=Rotate_Individual_Back(Mat,size,keys[n]);

			cv::eigen2cv(Mat, temp);
			temp=temp.reshape(chs,0);
			RGBChannels[n]=temp;
		}
		std::vector <Mat> channels;
		channels.push_back(RGBChannels[0]);
		channels.push_back(RGBChannels[1]);
		channels.push_back(RGBChannels[2]);
		cv::merge(channels,Image);
		std::cout<<keys[0]<<" "<<keys[1]<<" "<<keys[2]<<std::endl;
	}
	return Image;
}
Mat Matrix_Rotate_Back(Mat Image,int size,std::vector<std::vector <int> >& Keys)
{
	int rows=Image.rows;
	int cols=Image.cols;
	int i;
	int j;
	int row_end=0;
	int col_end=0;

	Mat Resultant_Image=Image.clone();
	Mat Reduced_Image;

	int box_count=(cols/size);
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
			int index;
			index=(i/size)*(box_count)+(j/size);
			Reduced_Image=Image(Range(i,row_end),Range(j,col_end));		
			Reduced_Image=Rotate_Sub_Matrix_Back(Reduced_Image,size,Keys[index]);						
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
	std::vector<std::vector <int> > Keys;
	Image=Matrix_Rotate(Image,10,Keys);
	imwrite("Encrypted_Image.jpg",Image);
	Image=Matrix_Rotate_Back(Image,10,Keys);
	imwrite("Getting_It_Back.jpg",Image);
	cv::waitKey(0);
	return 0;
}
