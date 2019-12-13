#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <ctime>
#include <vector>
#include <cmath>
using namespace cv;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd;

MatrixXd SVD_Individual(MatrixXd Matrix,int size)
{	

	MatrixXd temp;
	MatrixXd Transpose_Multi;
	temp=Matrix.transpose();
	Transpose_Multi=temp*Matrix;	

	Eigen::EigenSolver<MatrixXd> eigensolver;
    	eigensolver.compute(Transpose_Multi);
    	Eigen::MatrixXd eigen_vectors = eigensolver.eigenvectors().real();
	
	int k=2;
	if(k>size)
		return Matrix;
	std::cout<<Matrix<<std::endl<<std::endl;
	MatrixXd Sigma(k,k);
	int i,j;

	for(i=0;i<k;i++)
		for(j=0;j<k;j++)
		{
			if(i==j)
				Sigma(i,i)=sqrt(Transpose_Multi.eigenvalues().real()[i]);
			else
				Sigma(i,j)=0;
		}
	
	MatrixXd VMatrix(size,k);	
	
	for(i=0;i<k;i++)
		for(j=0;j<size;j++)
			VMatrix(j,i)=eigen_vectors(j,i);
	
	MatrixXd VMatrixT=VMatrix.transpose();

	MatrixXd UMatrix(size,k);
	MatrixXd Temp_Multi(1,size);
	MatrixXd Result_Multi(size,1);
	

	for(i=0;i<k;i++)
	{
		for(j=0;j<size;j++)
			Temp_Multi(0,j)=VMatrixT(i,j);
			Result_Multi=(Matrix*Temp_Multi.transpose())/Sigma(i,i);
	
		for(j=0;j<size;j++)
			UMatrix(j,i)=Result_Multi(j,0); 
	}
	MatrixXd Result=UMatrix*Sigma*VMatrixT;
	for(i=0;i<size;i++)
	{
		for(j=0;j<size;j++)
			if(Result(i,j)<0)
				return Matrix; 
	}
	std::cout<<Result<<std::endl<<std::endl;	
	return Result;
}

Mat SVD_Sub_Matrix(Mat Image,int size)
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

			Mat=SVD_Individual(Mat,size);
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

Mat SVDx(Mat Image,int size)
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
			continue;
		col_end=0;
		for(;col_end<cols;)
		{

			j=col_end;
			col_end+=size;
			if(col_end>cols)
				continue;
			Reduced_Image=Image(Range(i,row_end),Range(j,col_end));		
			Reduced_Image=SVD_Sub_Matrix(Reduced_Image,size);						
			Reduced_Image.copyTo(Resultant_Image.rowRange(i,row_end).colRange(j, col_end));	
		}	
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
	Image=SVDx(Image,10);
	imwrite("Decrease_size.jpg",Image);
	cv::waitKey(0);
	return 0;
}

