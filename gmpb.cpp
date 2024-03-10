/// Author: Vladimir Stanovov (vladimirstanovov@yandex.ru)
/// Last edited: March 10th, 2024
/// C++ implementation of the Generalized Moving Peaks Benchmark (GMPB)
/// Includes implementation of the mQSO algorithm (T. Blackwell and J. Branke, “Multiswarms, exclusion, and anticonvergence in dynamic environments,” IEEE Transactions on Evolutionary Computation, vol. 10, no. 4, pp. 459–472, 2006.)
/// This benchmark is a part of the CEC 2022 and 2024 competitions: https://danialyazdani.com/CEC-2022.php https://competition-hub.github.io/GMPB-Competition/
/// Reference:
/// D. Yazdani, M. N. Omidvar, R. Cheng, J. Branke, T. T. Nguyen, and X. Yao, “Benchmarking continuous dynamic optimization: Survey and generalized test suite,” IEEE Transactions on Cybernetics, vol. 52(5), pp. 3380-3393, 2020.
/// [2] M. Peng, Z. She, D. Yazdani, D. Yazdani, W. Luo, C. Li, J. Branke, T. T. Nguyen, A. H. Gandomi, Y. Jin, and X. Yao, “Evolutionary dynamic optimization laboratory: A matlab optimization platform for education and experimentation in dynamic environments,” arXiv preprint arXiv:2308.12644, 2023.
/// Original Evolutionary Dynamic Optimization Laboratory (EDOLAB): https://github.com/Danial-Yazdani/EDOLAB-MATLAB
#include <math.h>
#include <iostream>
#include <time.h>
#include <fstream>
#include <random>
#include <cstring> // for memset
#define PI 3.1415926535897932384626433832795029
using namespace std;
unsigned globalseed = unsigned(time(NULL));//2018;//
unsigned seed1 = globalseed+0;
unsigned seed2 = globalseed+100;
unsigned seed3 = globalseed+200;
unsigned seed4 = globalseed+300;
unsigned seed5 = globalseed+400;
std::mt19937 generator_uni_i(seed1);
std::mt19937 generator_uni_r(seed2);
std::mt19937 generator_norm(seed3);
std::mt19937 generator_cachy(seed4);
std::mt19937 generator_uni_i_2(seed5);
std::uniform_int_distribution<int> uni_int(0,32768);
std::uniform_real_distribution<double> uni_real(0.0,1.0);
std::normal_distribution<double> norm_dist(0.0,1.0);
std::cauchy_distribution<double> cachy_dist(0.0,1.0);

int IntRandom(int target)
{
    if(target == 0)
        return 0;
    return uni_int(generator_uni_i)%target;
}
double Random(double minimal, double maximal)
{
    return uni_real(generator_uni_r)*(maximal-minimal)+minimal;
}
double NormRand(double mu, double sigma)
{
    return norm_dist(generator_norm)*sigma + mu;
}
double CachyRand(double mu, double sigma)
{
    return cachy_dist(generator_cachy)*sigma+mu;
}

//QR decomposition code taken from https://rosettacode.org/wiki/QR_decomposition
class Vector;

class Matrix {

public:
  // default constructor (don't allocate)
  Matrix() : m(0), n(0), data(nullptr) {}

  // constructor with memory allocation, initialized to zero
  Matrix(int m_, int n_) : Matrix() {
    m = m_;
    n = n_;
    allocate(m_,n_);
  }

  Matrix(double** in2, int m_, int n_) : Matrix() {
    m = m_;
    n = n_;
    allocate(m_,n_);
    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
	(*this)(i,j) = in2[i][j];
  }

  // copy constructor
  Matrix(const Matrix& mat) : Matrix(mat.m,mat.n) {

    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
	(*this)(i,j) = mat(i,j);
  }

  // constructor from array
  template<int rows, int cols>
  Matrix(double (&a)[rows][cols]) : Matrix(rows,cols) {

    for (int i = 0; i < m; i++)
      for (int j = 0; j < n; j++)
	(*this)(i,j) = a[i][j];
  }

  // destructor
  ~Matrix() {
    deallocate();
  }


  // access data operators
  double& operator() (int i, int j) {
    return data[i+m*j]; }
  double  operator() (int i, int j) const {
    return data[i+m*j]; }

  // operator assignment
  Matrix& operator=(const Matrix& source) {

    // self-assignment check
    if (this != &source) {
      if ( (m*n) != (source.m * source.n) ) { // storage cannot be reused
	allocate(source.m,source.n);          // re-allocate storage
      }
      // storage can be used, copy data
      std::copy(source.data, source.data + source.m*source.n, data);
    }
    return *this;
  }

  // compute minor
  void compute_minor(const Matrix& mat, int d) {

    allocate(mat.m, mat.n);

    for (int i = 0; i < d; i++)
      (*this)(i,i) = 1.0;
    for (int i = d; i < mat.m; i++)
      for (int j = d; j < mat.n; j++)
	(*this)(i,j) = mat(i,j);

  }

  // Matrix multiplication
  // c = a * b
  // c will be re-allocated here
  void mult(const Matrix& a, const Matrix& b) {

    if (a.n != b.m) {
      std::cerr << "Matrix multiplication not possible, sizes don't match !\n";
      return;
    }

    // reallocate ourself if necessary i.e. current Matrix has not valid sizes
    if (a.m != m or b.n != n)
      allocate(a.m, b.n);

    memset(data,0,m*n*sizeof(double));

    for (int i = 0; i < a.m; i++)
      for (int j = 0; j < b.n; j++)
	for (int k = 0; k < a.n; k++)
	  (*this)(i,j) += a(i,k) * b(k,j);

  }

  void transpose() {
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < i; j++) {
	double t = (*this)(i,j);
	(*this)(i,j) = (*this)(j,i);
	(*this)(j,i) = t;
      }
    }
  }

  // take c-th column of m, put in v
  void extract_column(Vector& v, int c);

  // memory allocation
  void allocate(int m_, int n_) {

    // if already allocated, memory is freed
    deallocate();

    // new sizes
    m = m_;
    n = n_;

    data = new double[m_*n_];
    memset(data,0,m_*n_*sizeof(double));

  } // allocate

  // memory free
  void deallocate() {

    if (data)
      delete[] data;

    data = nullptr;

  }

  int m, n;

private:
  double* data;

}; // struct Matrix

// column vector
class Vector {

public:
  // default constructor (don't allocate)
  Vector() : size(0), data(nullptr) {}

  // constructor with memory allocation, initialized to zero
  Vector(int size_) : Vector() {
    size = size_;
    allocate(size_);
  }

  // destructor
  ~Vector() {
    deallocate();
  }

  // access data operators
  double& operator() (int i) {
    return data[i]; }
  double  operator() (int i) const {
    return data[i]; }

  // operator assignment
  Vector& operator=(const Vector& source) {

    // self-assignment check
    if (this != &source) {
      if ( size != (source.size) ) {   // storage cannot be reused
	allocate(source.size);         // re-allocate storage
      }
      // storage can be used, copy data
      std::copy(source.data, source.data + source.size, data);
    }
    return *this;
  }

  // memory allocation
  void allocate(int size_) {

    deallocate();

    // new sizes
    size = size_;

    data = new double[size_];
    memset(data,0,size_*sizeof(double));

  } // allocate

  // memory free
  void deallocate() {

    if (data)
      delete[] data;

    data = nullptr;

  }

  //   ||x||
  double norm() {
    double sum = 0;
    for (int i = 0; i < size; i++) sum += (*this)(i) * (*this)(i);
    return sqrt(sum);
  }

  // divide data by factor
  void rescale(double factor) {
    for (int i = 0; i < size; i++) (*this)(i) /= factor;
  }

  void rescale_unit() {
    double factor = norm();
    rescale(factor);
  }

  int size;

private:
  double* data;

}; // class Vector

// c = a + b * s
void vmadd(const Vector& a, const Vector& b, double s, Vector& c)
{
  if (c.size != a.size or c.size != b.size) {
    std::cerr << "[vmadd]: vector sizes don't match\n";
    return;
  }

  for (int i = 0; i < c.size; i++)
    c(i) = a(i) + s * b(i);
}

// mat = I - 2*v*v^T
// !!! m is allocated here !!!
void compute_householder_factor(Matrix& mat, const Vector& v)
{

  int n = v.size;
  mat.allocate(n,n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      mat(i,j) = -2 *  v(i) * v(j);
  for (int i = 0; i < n; i++)
    mat(i,i) += 1;
}

// take c-th column of a matrix, put results in Vector v
void Matrix::extract_column(Vector& v, int c) {
  if (m != v.size) {
    std::cerr << "[Matrix::extract_column]: Matrix and Vector sizes don't match\n";
    return;
  }

  for (int i = 0; i < m; i++)
    v(i) = (*this)(i,c);
}

void matrix_show(const Matrix&  m, const std::string& str="")
{
  std::cout << str << "\n";
  for(int i = 0; i < m.m; i++) {
    for (int j = 0; j < m.n; j++) {
      printf(" %8.3f", m(i,j));
    }
    printf("\n");
  }
  printf("\n");
}

// L2-norm ||A-B||^2
double matrix_compare(const Matrix& A, const Matrix& B) {
  // matrices must have same size
  if (A.m != B.m or  A.n != B.n)
    return std::numeric_limits<double>::max();

  double res=0;
  for(int i = 0; i < A.m; i++) {
    for (int j = 0; j < A.n; j++) {
      res += (A(i,j)-B(i,j)) * (A(i,j)-B(i,j));
    }
  }

  res /= A.m*A.n;
  return res;
}

void householder(Matrix& mat,
		 Matrix& R,
		 Matrix& Q)
{

  int m = mat.m;
  int n = mat.n;

  // array of factor Q1, Q2, ... Qm
  std::vector<Matrix> qv(m);

  // temp array
  Matrix z(mat);
  Matrix z1;

  for (int k = 0; k < n && k < m - 1; k++) {

    Vector e(m), x(m);
    double a;

    // compute minor
    z1.compute_minor(z, k);

    // extract k-th column into x
    z1.extract_column(x, k);

    a = x.norm();
    if (mat(k,k) > 0) a = -a;

    for (int i = 0; i < e.size; i++)
      e(i) = (i == k) ? 1 : 0;

    // e = x + a*e
    vmadd(x, e, a, e);

    // e = e / ||e||
    e.rescale_unit();

    // qv[k] = I - 2 *e*e^T
    compute_householder_factor(qv[k], e);

    // z = qv[k] * z1
    z.mult(qv[k], z1);

  }

  Q = qv[0];

  // after this loop, we will obtain Q (up to a transpose operation)
  for (int i = 1; i < n && i < m - 1; i++) {

    z1.mult(qv[i], Q);
    Q = z1;

  }

  R.mult(Q, mat);
  Q.transpose();
}

double Edist(double* x, double* y, int n)
{
    double res = 0;
    for(int i=0;i!=n;i++)
        res += (x[i]-y[i])*(x[i]-y[i]);
    return sqrt(res);
}


void matmul(double** matrix, double* vec, double* out, const int dim)
{
    for(int i=0;i!=dim;i++)
    {
        out[i] = 0;
        for(int j=0;j!=dim;j++)
            out[i] += matrix[i][j]*vec[j];
    }
}
void matmul2(double** matrix, double* vec, double* out, const int dim)
{
    for(int i=0;i!=dim;i++)
    {
        out[i] = 0;
        for(int j=0;j!=dim;j++)
            out[i] += matrix[j][i]*vec[j];
    }
}
void matmul3(double** matrix1, double** matrix2, double** matrix3, const int dim)
{
    for(int i=0;i!=dim;i++)
    {
        for(int j=0;j!=dim;j++)
        {
            matrix3[i][j] = 0;
            for(int k=0;k!=dim;k++)
                matrix3[i][j] += matrix1[i][k]*matrix2[k][j];
        }
    }
}
void Rotation(double teta, int dim, double** output2, double*** X, double** output, int* randperm, int PageNumber)
{
    int counter = 0;
    for(int i=0;i!=dim;i++)
    {
        for(int j=i+1;j!=dim;j++)
        {
            if(i!=j)
            {
                for(int k=0;k!=dim;k++)
                {
                    for(int L=0;L!=dim;L++)
                        X[counter][k][L] = 0;
                    X[counter][k][k] = 1;
                }
                X[counter][i][i] = cos(teta);
                X[counter][j][j] = cos(teta);
                X[counter][i][j] = sin(teta);
                X[counter][j][i] =-sin(teta);
                counter++;
            }
        }
    }
    counter=0;
    for(int i=0;i!=dim;i++)
    {
        for(int j=0;j!=dim;j++)
            output[i][j] = 0;
        output[i][i] = 1;
    }
    for(int i=0;i!=PageNumber;i++)
        randperm[i] = i;
    for(int i=0;i!=PageNumber*5;i++)
        swap(randperm[IntRandom(PageNumber)],randperm[IntRandom(PageNumber)]);
    for(int i=0;i!=PageNumber;i++)
    {
        matmul3(output,X[randperm[i]],output2,dim);
        for(int j=0;j!=dim;j++)
        {
            for(int k=0;k!=dim;k++)
                output[j][k] = output2[j][k];
        }
    }
    for(int i=0;i!=dim;i++)
    {
        for(int j=0;j!=dim;j++)
            output2[i][j] = output[i][j];
    }
}
class GMPB
{
public:
    int PeakNumber;
    int Dimension;
    int FEval;
    int ChangeFrequency;
    int EnvironmentNumber;
    int EnvironmentCounter;
    int RecentChange;
    int MaxEvals;
    int PageNumber;
    int* randperm;
    double ShiftSeverity;
    double* Ebbc;
    double* CurrentError;
    double* CurrentPerformance;
    double MinCoordinate = -100;
    double MaxCoordinate = 100;
    double MinHeight = 30;
    double MaxHeight = 70;
    double MinWidth = 1;
    double MaxWidth = 12;
    double MinAngle = -PI;
    double MaxAngle = PI;
    double MinTau = -1;
    double MaxTau = 1;
    double MinEta = -20;
    double MaxEta = 20;
    double HeightSeverity = 7;
    double* OptimumValue;
    double* OptimumID;
    double** PeakVisibility;
    double** PeaksHeight;
    double*** PeaksPosition;
    double*** PeaksWidth;
    double AngleSeverity = PI/9.0;
    double TauSeverity = 0.2;
    double EtaSeverity = 2;
    double EllipticalPeaks = 1;
    double*** InitialRotationMatrix;
    double WidthSeverity = 1;
    double** PeaksAngle;
    double** tau;
    double** ShiftOffset;
    double*** eta;
    double* fval;
    double**** RotationMatrix;
    double** output;
    double** output2;
    double*** X;
    double* temp;
    double* temp2;
    double* a;
    double* b;
    GMPB(){};
    void Init(int newPeakNumber,
                int newDimension,
                int newChangeFrequency,
                int newShiftSeverity,
                int newEnvironmentNumber);
    void Clear();
    ~GMPB(){};
    double Fitness(double* xvec);
};
void GMPB::Init(int newPeakNumber,
                int newDimension,
                int newChangeFrequency,
                int newShiftSeverity,
                int newEnvironmentNumber)
{
    PeakNumber = newPeakNumber;
    Dimension = newDimension;
    ChangeFrequency = newChangeFrequency;
    ShiftSeverity = newShiftSeverity;
    EnvironmentNumber = newEnvironmentNumber;
    EnvironmentCounter = 0;
    RecentChange = 0;
    FEval = 0;
    MaxEvals = ChangeFrequency*EnvironmentNumber;
    Ebbc = new double[EnvironmentNumber];
    CurrentError = new double[MaxEvals];
    CurrentPerformance = new double[MaxEvals];
    OptimumValue = new double[EnvironmentNumber];
    OptimumID = new double[EnvironmentNumber];
    PeakVisibility = new double*[EnvironmentNumber];
    PeaksHeight = new double*[EnvironmentNumber];
    PeaksPosition = new double**[EnvironmentNumber];
    PeaksWidth = new double**[EnvironmentNumber];
    eta = new double**[EnvironmentNumber];
    PeaksAngle = new double*[EnvironmentNumber];
    tau = new double*[EnvironmentNumber];
    RotationMatrix = new double***[EnvironmentNumber];
    for(int i=0;i!=EnvironmentNumber;i++)
    {
        PeakVisibility[i] = new double[PeakNumber];
        PeaksHeight[i] = new double[PeakNumber];
        PeaksPosition[i] = new double*[PeakNumber];
        PeaksWidth[i] = new double*[PeakNumber];
        eta[i] = new double*[PeakNumber];
        PeaksAngle[i] = new double[PeakNumber];
        tau[i] = new double[PeakNumber];
        RotationMatrix[i] = new double**[PeakNumber];
        for(int j=0;j!=PeakNumber;j++)
        {
            PeaksPosition[i][j] = new double[Dimension];
            PeaksWidth[i][j] = new double[Dimension];
            eta[i][j] = new double[4];
            RotationMatrix[i][j] = new double*[Dimension];
            for(int k=0;k!=Dimension;k++)
            {
                RotationMatrix[i][j][k] = new double[Dimension];
            }
        }
    }
    output2 = new double*[Dimension];
    for(int i=0;i!=Dimension;i++)
    {
        output2[i] = new double[Dimension];
    }
    PageNumber = double(Dimension)*((double(Dimension)-1.)*0.5);
    X = new double**[PageNumber];
    for(int i=0;i!=PageNumber;i++)
    {
        X[i] = new double*[Dimension];
        for(int j=0;j!=Dimension;j++)
            X[i][j] = new double[Dimension];
    }
    output = new double*[Dimension];
    for(int i=0;i!=Dimension;i++)
    {
        output[i] = new double[Dimension];
        for(int j=0;j!=Dimension;j++)
            output[i][j] = 0;
        output[i][i] = 1;
    }
    randperm = new int[PageNumber];
    for(int j=0;j!=PeakNumber;j++)
    {
        for(int k=0;k!=Dimension;k++)
        {
            PeaksPosition[0][j][k] = Random(MinCoordinate,MaxCoordinate);
            PeaksWidth[0][j][k] = Random(MinWidth,MaxWidth);
        }
        PeaksHeight[0][j] = Random(MinHeight,MaxHeight);
        PeaksAngle[0][j] = Random(MinAngle,MaxAngle);
        if(j == 0)
        {
            OptimumValue[0] = PeaksHeight[0][j];
            OptimumID[0] = j;
        }
        else if(OptimumValue[0] < PeaksHeight[0][j])
        {
            OptimumValue[0] = PeaksHeight[0][j];
            OptimumID[0] = j;
        }
        tau[0][j] = Random(MinTau,MaxTau);
        for(int k=0;k!=4;k++)
        {
            eta[0][j][k] = Random(MinEta,MaxEta);
        }
        PeakVisibility[0][j] = 1;
    }
    InitialRotationMatrix = new double**[PeakNumber];
    for(int i=0;i!=PeakNumber;i++)
    {
        InitialRotationMatrix[i] = new double*[Dimension];
        for(int j=0;j!=Dimension;j++)
            InitialRotationMatrix[i][j] = new double[Dimension];
        for(int j=0;j!=Dimension;j++)
            for(int k=0;k!=Dimension;k++)
                InitialRotationMatrix[i][j][k] = Random(0,1);
        Matrix A(InitialRotationMatrix[i],Dimension,Dimension);
        Matrix Q, R;
        householder(A, R, Q);
        for(int j=0;j!=Dimension;j++)
        {
            for(int k=0;k!=Dimension;k++)
            {
                InitialRotationMatrix[i][j][k] = Q(j,k);
                /*InitialRotationMatrix[i][j][k] = 0;
                if(j == k)
                    InitialRotationMatrix[i][j][j] = 1;*/
                RotationMatrix[0][i][j][k] = InitialRotationMatrix[i][j][k];
            }
        }
    }
    ShiftOffset = new double*[PeakNumber];
    for(int i=0;i!=PeakNumber;i++)
    {
        ShiftOffset[i] = new double[Dimension];
    }
    for(int e=1;e!=EnvironmentNumber;e++)
    {
        for(int i=0;i!=PeakNumber;i++)
        {
            double dist = 0;
            for(int j=0;j!=Dimension;j++)
            {
                ShiftOffset[i][j] = NormRand(0,1);
                dist += (ShiftOffset[i][j])*(ShiftOffset[i][j]);
            }
            dist = sqrt(dist);
            for(int j=0;j!=Dimension;j++)
            {
                PeaksPosition[e][i][j] = PeaksPosition[e-1][i][j] + ShiftOffset[i][j]/dist*ShiftSeverity;
                if(PeaksPosition[e][i][j] > MaxCoordinate)
                    PeaksPosition[e][i][j] = 2*MaxCoordinate - PeaksPosition[e][i][j];
                if(PeaksPosition[e][i][j] < MinCoordinate)
                    PeaksPosition[e][i][j] = 2*MinCoordinate - PeaksPosition[e][i][j];
                //PeaksPosition[e][i][j] = PeaksPosition[e][i][j]*(PeaksPosition[e][i][j] < MaxCoordinate) + (2*MinCoordinate - PeaksPosition[e][i][j])*(PeaksPosition[e][i][j] > MaxCoordinate);
                //PeaksPosition[e][i][j] = PeaksPosition[e][i][j]*(PeaksPosition[e][i][j] > MinCoordinate) + (2*MinCoordinate - PeaksPosition[e][i][j])*(PeaksPosition[e][i][j] < MinCoordinate);
                //do
                //{
                    PeaksWidth[e][i][j] = PeaksWidth[e-1][i][j] + NormRand(0,1)*WidthSeverity;
                //} while(PeaksWidth[e][i][j] > MaxWidth || PeaksWidth[e][i][j] < MinWidth);
                if(PeaksWidth[e][i][j] > MaxWidth)
                    PeaksWidth[e][i][j] = 2*MaxWidth - PeaksWidth[e][i][j];
                if(PeaksWidth[e][i][j] < MinWidth)
                    PeaksWidth[e][i][j] = 2*MinWidth - PeaksWidth[e][i][j];
            }
            //do
            //{
                PeaksHeight[e][i] = PeaksHeight[e-1][i] + NormRand(0,1)*HeightSeverity;
            //} while(PeaksHeight[e][i] > MaxHeight || PeaksHeight[e][i] < MinHeight);
            if(PeaksHeight[e][i] > MaxHeight)
                PeaksHeight[e][i] = 2*MaxHeight - PeaksHeight[e][i];
            if(PeaksHeight[e][i] < MinHeight)
                PeaksHeight[e][i] = 2*MinHeight - PeaksHeight[e][i];
            //do
            //{
                PeaksAngle[e][i] = PeaksAngle[e-1][i] + NormRand(0,1)*AngleSeverity;
            //} while(PeaksAngle[e][i] > MaxAngle || PeaksAngle[e][i] < MinAngle);
            if(PeaksAngle[e][i] > MaxAngle)
                PeaksAngle[e][i] = 2*MaxAngle - PeaksAngle[e][i];
            if(PeaksAngle[e][i] < MinAngle)
                PeaksAngle[e][i] = 2*MinAngle - PeaksAngle[e][i];
            //do
            //{
                tau[e][i] = tau[e-1][i] + NormRand(0,1)*TauSeverity;
            //} while(tau[e][i] > MaxTau || tau[e][i] < MinTau);
            if(tau[e][i] > MaxTau)
                tau[e][i] = 2*MaxTau - tau[e][i];
            if(tau[e][i] < MinTau)
                tau[e][i] = 2*MinTau - tau[e][i];
            for(int j=0;j!=4;j++)
            {
                //do
                //{
                    eta[e][i][j] = eta[e-1][i][j] + NormRand(0,1)*EtaSeverity;
                //} while(eta[e][i][j] > MaxEta || eta[e][i][j] < MinEta);
                if(eta[e][i][j] > MaxEta)
                    eta[e][i][j] = 2*MaxEta - eta[e][i][j];
                if(eta[e][i][j] < MinEta)
                    eta[e][i][j] = 2*MinEta - eta[e][i][j];
            }
            Rotation(PeaksAngle[e][i],Dimension,output2,X,output,randperm,PageNumber);
            matmul3(InitialRotationMatrix[i],output2,RotationMatrix[e][i],Dimension);
            if(i == 0)
            {
                OptimumValue[e] = PeaksHeight[e][i];
                OptimumID[e] = i;
            }
            else if(OptimumValue[e] < PeaksHeight[e][i])
            {
                OptimumValue[e] = PeaksHeight[e][i];
                OptimumID[e] = i;
            }
            PeakVisibility[e][i] = 1;
        }
    }
    EnvironmentCounter = 0;

    fval = new double[PeakNumber];
    temp = new double[Dimension];
    temp2 = new double[Dimension];
    a = new double[Dimension];
    b = new double[Dimension];
}
double GMPB::Fitness(double* xvec)
{
    double res = 0;
    for(int i=0;i!=PeakNumber;i++)
    {
        for(int j=0;j!=Dimension;j++)
            temp2[j] = xvec[j] - PeaksPosition[EnvironmentCounter][i][j];
        matmul(RotationMatrix[EnvironmentCounter][i],temp2,temp,Dimension);
        for(int j=0;j!=Dimension;j++)
        {
            if(temp[j] > 0)
                a[j] = exp(log( temp[j])+tau[EnvironmentCounter][i]*(sin(eta[EnvironmentCounter][i][0]*log( temp[j]))+sin(eta[EnvironmentCounter][i][1]*log( temp[j]))));
            else if(temp[j] < 0)
                a[j] =-exp(log(-temp[j])+tau[EnvironmentCounter][i]*(sin(eta[EnvironmentCounter][i][2]*log(-temp[j]))+sin(eta[EnvironmentCounter][i][3]*log(-temp[j]))));
            else
                a[j] = 0;
        }
        fval[i] = 0;
        for(int j=0;j!=Dimension;j++)
        {
            fval[i] += a[j]*a[j]*PeaksWidth[EnvironmentCounter][i][j]*PeaksWidth[EnvironmentCounter][i][j];
        }
        fval[i] = PeaksHeight[EnvironmentCounter][i] - sqrt(fval[i]);
        if(i == 0)
            res = fval[i];
        else
            res = max(res,fval[i]);
    }
    if(FEval > MaxEvals)
    {
        return res;
    }
    double SolutionError = OptimumValue[EnvironmentCounter] - res;
    if(FEval%ChangeFrequency != 0)
    {
        if(CurrentError[FEval-1] < SolutionError)
        {
            CurrentError[FEval] = CurrentError[FEval-1];
            CurrentPerformance[FEval] = CurrentPerformance[FEval-1];
        }
        else
        {
            CurrentError[FEval] = SolutionError;
            CurrentPerformance[FEval] = res;
        }
    }
    else
    {
        CurrentError[FEval] = SolutionError;
        CurrentPerformance[FEval] = res;
    }
    if(FEval%ChangeFrequency == ChangeFrequency-1)
    {
        Ebbc[EnvironmentCounter] = CurrentError[FEval];
    }
    FEval++;
    if(FEval%ChangeFrequency == 0 && FEval < MaxEvals)
    {
        EnvironmentCounter++;
        RecentChange = 1;
    }
    else
    {
        //RecentChange = 0;
    }
    return res;
}
void GMPB::Clear()
{
    delete Ebbc;
    delete CurrentError;
    delete CurrentPerformance;
    delete OptimumValue;
    delete OptimumID;
    for(int i=0;i!=EnvironmentNumber;i++)
    {
        delete PeakVisibility[i];
        delete PeaksHeight[i];
        for(int j=0;j!=PeakNumber;j++)
        {
            delete PeaksPosition[i][j];
            delete PeaksWidth[i][j];
            for(int k=0;k!=Dimension;k++)
            {
                delete RotationMatrix[i][j][k];
            }
            delete eta[i][j];
            delete RotationMatrix[i][j];
        }
        delete PeaksPosition[i];
        delete PeaksWidth[i];
        delete eta[i];
        delete RotationMatrix[i];
        delete PeaksAngle[i];
        delete tau[i];
    }
    for(int j=0;j!=PeakNumber;j++)
    {
        for(int k=0;k!=Dimension;k++)
        {
            delete InitialRotationMatrix[j][k];
        }
        delete InitialRotationMatrix[j];
        delete ShiftOffset[j];
    }
    delete InitialRotationMatrix;
    delete ShiftOffset;
    delete PeakVisibility;
    delete PeaksHeight;
    delete PeaksPosition;
    delete PeaksWidth;
    delete eta;
    delete RotationMatrix;
    delete PeaksAngle;
    delete tau;
    for(int i=0;i!=Dimension;i++)
    {
        delete output2[i];
    }
    delete output2;
    for(int i=0;i!=PageNumber;i++)
    {
        for(int j=0;j!=Dimension;j++)
            delete X[i][j];
        delete X[i];
    }
    delete X;
    for(int i=0;i!=Dimension;i++)
        delete output[i];
    delete output;
    delete randperm;

    delete fval;
    delete temp;
    delete temp2;
    delete a;
    delete b;
}

class Optimizer //mQSO
{
public:
    int NInds;
    int NVars;
    int NSwarm;
    int QuantumNumber;
    int* IsConverged;
    int* BestID;
    double Left;
    double Right;
    double c1;
    double c2;
    double w;
    double QuantumRadius;
    double ExcLimit;
    double ConvLimit;
    double* BestVal;
    double* QuantumPos;
    double*** Pos;
    double*** Vel;
    double*** PBest;
    double** Fit;
    double** FBest;
    Optimizer(){};
    ~Optimizer(){};
    void Init(int newNInds,
              int newNVars,
              int newNSwarm,
              int newQuantumNumber,
              double newLeft,
              double newRight,
              GMPB& ngmpb);
    void Clear();
    void Step(GMPB& ngmpb);
    void Reaction(GMPB& ngmpb);
};
void Optimizer::Init(int newNInds,
                     int newNVars,
                     int newNSwarm,
                     int newQuantumNumber,
                     double newLeft,
                     double newRight,
                     GMPB& ngmpb)
{
    NInds = newNInds;
    NVars = newNVars;
    NSwarm = newNSwarm;
    QuantumNumber = newQuantumNumber;
    Left = newLeft;
    Right = newRight;
    QuantumRadius = 1;
    ExcLimit = 0.5*(Right-Left)/(pow(double(NSwarm),1.0/double(NVars)));
    ConvLimit = ExcLimit;
    c1 = 2.05;
    c2 = 2.05;
    w = 0.729843788;
    Pos = new double**[NSwarm];
    Vel = new double**[NSwarm];
    PBest = new double**[NSwarm];
    Fit = new double*[NSwarm];
    FBest = new double*[NSwarm];
    BestVal = new double[NSwarm];
    BestID = new int[NSwarm];
    IsConverged = new int[NSwarm];
    QuantumPos = new double[NVars];
    for(int i=0;i!=NSwarm;i++)
    {
        Pos[i] = new double*[NInds];
        Vel[i] = new double*[NInds];
        PBest[i] = new double*[NInds];
        Fit[i] = new double[NInds];
        FBest[i] = new double[NInds];
        for(int j=0;j!=NInds;j++)
        {
            Pos[i][j] = new double[NVars];
            Vel[i][j] = new double[NVars];
            PBest[i][j] = new double[NVars];
            for(int k=0;k!=NVars;k++)
            {
                Pos[i][j][k] = Random(Left,Right);
                Vel[i][j][k] = 0;
                PBest[i][j][k] = Pos[i][j][k];
            }
            Fit[i][j] = ngmpb.Fitness(Pos[i][j]);
            if(ngmpb.RecentChange)
            {
                Fit[i][j] = numeric_limits<double>::min();
            }
            FBest[i][j] = Fit[i][j];
            if(j == 0 || FBest[i][j] > BestVal[i])
            {
                BestVal[i] = FBest[i][j];
                BestID[i] = j;
            }
        }
    }
}
void Optimizer::Clear()
{
    for(int i=0;i!=NSwarm;i++)
    {
        for(int j=0;j!=NInds;j++)
        {
            delete Pos[i][j];
            delete Vel[i][j];
            delete PBest[i][j];
        }
        delete Pos[i];
        delete Vel[i];
        delete PBest[i];
        delete Fit[i];
        delete FBest[i];
    }
    delete Pos;
    delete Vel;
    delete PBest;
    delete Fit;
    delete FBest;
    delete BestVal;
    delete BestID;
    delete IsConverged;
    delete QuantumPos;
}
void Optimizer::Step(GMPB& ngmpb)
{
    for(int i=0;i!=NSwarm;i++)
    {
        for(int j=0;j!=NInds;j++)
        {
            for(int k=0;k!=NVars;k++)
            {
                Vel[i][j][k] = w*(Vel[i][j][k]
                   + c1*Random(0,1)*(PBest[i][j][k] - Pos[i][j][k])
                   + c2*Random(0,1)*(PBest[i][BestID[i]][k] - Pos[i][j][k]));
                Pos[i][j][k] += Vel[i][j][k];
                Vel[i][j][k] *= (Pos[i][j][k] < Right);
                Vel[i][j][k] *= (Pos[i][j][k] > Left);
                Pos[i][j][k] = min(Pos[i][j][k],Right);
                Pos[i][j][k] = max(Pos[i][j][k],Left);
            }
            double temp = ngmpb.Fitness(Pos[i][j]);
            if(ngmpb.RecentChange)
                return;
            Fit[i][j] = temp;
        }
        for(int j=0;j!=NInds;j++)
        {
            if(Fit[i][j] > FBest[i][j])
            {
                FBest[i][j] = Fit[i][j];
                for(int k=0;k!=NVars;k++)
                {
                    PBest[i][j][k] = Pos[i][j][k];
                }
                if(FBest[i][j] > BestVal[i])
                {
                    BestVal[i] = FBest[i][j];
                    BestID[i] = j;
                }
            }
        }
        for(int j=0;j!=QuantumNumber;j++)
        {
            for(int k=0;k!=NVars;k++)
            {
                QuantumPos[k] = PBest[i][BestID[i]][k] + Random(-QuantumRadius,QuantumRadius);
            }
            double QuantumFit = ngmpb.Fitness(QuantumPos);
            if(ngmpb.RecentChange)
                return;
            if(QuantumFit > BestVal[i])
            {
                BestVal[i] = QuantumFit;
                FBest[i][BestID[i]] = QuantumFit;
                for(int k=0;k!=NVars;k++)
                {
                    PBest[i][BestID[i]][k] = QuantumPos[k];
                }
            }
        }
    }
    //return;
    for(int i1=0;i1!=NSwarm-1;i1++)
    {
        for(int i2=i1+1;i2!=NSwarm;i2++)
        {
            double dist = Edist(PBest[i1][BestID[i1]],PBest[i2][BestID[i2]],NVars);
            if(dist < ExcLimit)
            {
                int index = i1;
                if(BestVal[i1] > BestVal[i2])
                    index = i2;
                for(int j=0;j!=NInds;j++)
                {
                    for(int k=0;k!=NVars;k++)
                    {
                        Pos[index][j][k] = Random(Left,Right);
                        Vel[index][j][k] = 0;
                        PBest[index][j][k] = Pos[index][j][k];
                    }
                    Fit[index][j] = ngmpb.Fitness(Pos[index][j]);
                    if(ngmpb.RecentChange)
                    {
                        Fit[index][j] = numeric_limits<double>::min();
                    }
                    FBest[index][j] = Fit[index][j];
                    if(j == 0 || FBest[index][j] > BestVal[index])
                    {
                        BestVal[index] = FBest[index][j];
                        BestID[index] = j;
                    }
                }
                if(ngmpb.RecentChange)
                    break;
            }
        }
    }
    int IsAllConverged = 0;
    double WorstVal = numeric_limits<double>::max();
    int WorstIndex = 0;
    double Radius;
    for(int i=0;i!=NSwarm;i++)
    {
        Radius = 0;
        for(int j=0;j!=NInds;j++)
        {
            for(int k=0;k!=NInds;k++)
            {
                for(int L=0;L!=NVars;L++)
                {
                    Radius = max(Radius,abs(Pos[i][j][L]-Pos[i][k][L]));
                }
            }
        }
        IsConverged[i] = (Radius < ConvLimit);
        IsAllConverged += IsConverged[i];
        if(BestVal[i] < WorstVal)
        {
            WorstVal = BestVal[i];
            WorstIndex = i;
        }
    }
    if(IsAllConverged == NSwarm)
    {
        for(int j=0;j!=NInds;j++)
        {
            for(int k=0;k!=NVars;k++)
            {
                Pos[WorstIndex][j][k] = Random(Left,Right);
                Vel[WorstIndex][j][k] = 0;
                PBest[WorstIndex][j][k] = Pos[WorstIndex][j][k];
            }
            Fit[WorstIndex][j] = ngmpb.Fitness(Pos[WorstIndex][j]);
            if(ngmpb.RecentChange == 1)
            {
                Fit[WorstIndex][j] = numeric_limits<double>::min();
            }
            FBest[WorstIndex][j] = Fit[WorstIndex][j];
            if(j == 0 || FBest[WorstIndex][j] > BestVal[WorstIndex])
            {
                BestVal[WorstIndex] = FBest[WorstIndex][j];
                BestID[WorstIndex] = j;
            }
        }
    }
}
void Optimizer::Reaction(GMPB& ngmpb)
{
    for(int i=0;i!=NSwarm;i++)
    {
        for(int j=0;j!=NInds;j++)
        {
            FBest[i][j] = ngmpb.Fitness(PBest[i][j]);
            if(j == 0 || FBest[i][j] > BestVal[i])
            {
                BestVal[i] = FBest[i][j];
                BestID[i] = j;
            }
        }
        IsConverged[i] = 0;
    }
}
int main()
{
    int mode = 1; //0 - generate for visualization; 1 - run mQSO algorithm
    if(mode == 0)
    {
        double step = 2;
        int N = 101;
        double** fvals = new double*[N];
        for(int i=0;i!=N;i++)
            fvals[i] = new double[N];
        GMPB ngm;
        ngm.Init(3,2,10201,1,100);
        char buffer[100];
        double* xvec = new double[ngm.Dimension];
        for(int e=0;e!=100;e++)
        {
            sprintf(buffer,"res_e%d.txt",e);
            ofstream fout(buffer);
            for(int i=0;i!=N;i++)
            {
                for(int j=0;j!=N;j++)
                {
                    xvec[0] = ngm.MinCoordinate + step*i;
                    xvec[1] = ngm.MinCoordinate + step*j;
                    fvals[i][j] = ngm.Fitness(xvec);
                    fout<<fvals[i][j]<<"\t";
                }
                fout<<endl;
            }
        }
        ofstream fout2("CurrentError.txt");
        for(int i=0;i!=10201*100;i++)
            fout2<<ngm.CurrentError[i]<<"\t";
        ngm.Clear();
        delete xvec;
        for(int i=0;i!=N;i++)
            delete fvals[i];
        delete fvals;
    }
    else
    {
        for(int run=0;run!=31;run++)
        {
            double lastNFEval = 0;
            GMPB ngmpb;
            ngmpb.Init(10,5,5000,1,100);
            Optimizer Opt;
            Opt.Init(5,ngmpb.Dimension,10,5,ngmpb.MinCoordinate,ngmpb.MaxCoordinate,ngmpb);
            while(ngmpb.FEval < ngmpb.MaxEvals)
            {
                Opt.Step(ngmpb);
                if(ngmpb.RecentChange)
                {
                    ngmpb.RecentChange = 0;
                    Opt.Reaction(ngmpb);
                }
                /*cout<<ngmpb.FEval<<"\t";
                for(int i=0;i!=Opt.NSwarm;i++)
                {
                    cout<<Opt.BestVal[i]<<"\t";
                }
                cout<<endl;*/
                if(ngmpb.FEval > lastNFEval)
                {
                    cout<<"Run: "<<run<<"\tFE: "<<ngmpb.FEval<<"\tCurrentError: "<<ngmpb.CurrentError[ngmpb.FEval-1]<<endl;
                    lastNFEval = ngmpb.FEval + 10000;
                }
            }
            char buffer[100];
            sprintf(buffer,"CurrentError_r%d.txt",run);
            ofstream fout2(buffer);
            for(int i=0;i!=ngmpb.MaxEvals;i++)
                fout2<<ngmpb.CurrentError[i]<<"\n";
            sprintf(buffer,"CurrentPerformance_r%d.txt",run);
            ofstream fout3(buffer);
            for(int i=0;i!=ngmpb.MaxEvals;i++)
                fout3<<ngmpb.CurrentPerformance[i]<<"\n";
            sprintf(buffer,"Ebbc_r%d.txt",run);
            ofstream fout4(buffer);
            for(int i=0;i!=ngmpb.EnvironmentNumber;i++)
                fout4<<ngmpb.Ebbc[i]<<"\n";
            Opt.Clear();
            ngmpb.Clear();
        }
    }
    return 0;
}
