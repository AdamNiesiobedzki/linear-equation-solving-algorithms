#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>

#define N 941
#define f 8
#define residualError 10e-9
//index 188641
using namespace std;

struct iterationData
{
    int matrixSize;
    double time;
};

void backwardSubstitution(double* output, double** matrix, double* vector, int size)
{
    for (int i = size - 1; i >= 0; i--)
    {
        double temp = 0.0;
        for (int j = i + 1; j < size; j++)
        {
            temp += matrix[i][j] * output[j];
        }
        output[i] = (vector[i] - temp) / matrix[i][i];
    }
}


void forwardSubstitution(double* output, double** matrix, double* vector, int size)
{
    for (int i = 0; i < size; i++)
    {
        double temp = 0.0;
        for (int j = 0; j < i; j++)
        {
            temp += matrix[i][j] * output[j];
        }
        output[i] = (vector[i] - temp) / matrix[i][i];
    }
}


void factorizationLU(double** matrix, double* b, int size)
{

    double** low = new double* [size];
    double** up = new double* [size];
    double* x = new double[size];
    double* y = new double[size];
    double* res = new double[size];

    //init
    //macierz low jako macierz jednostkowa
    //macierz up to przekopiowana macierz źródłowa
    for (int i = 0; i < size; i++)
    {
        low[i] = new double[size];
        up[i] = new double[size];
        for (int j = 0; j < size; j++)
        {
            if (i == j)
                low[i][j] = 1.0;
            else
                low[i][j] = 0.0;

            up[i][j] = matrix[i][j];
        }
    }

    //faktoryzacja
    // L * U = M
    for (int i = 0; i < size; i++)
    {
        for (int j = i + 1; j < size; j++)
        {
            low[j][i] = up[j][i] / up[i][i];

            for (int k = i; k < size; k++)
            {
                up[j][k] -= low[j][i] * up[i][k];
            }
        }
    }

    forwardSubstitution(y, low, b, size);

    backwardSubstitution(x, up, y, size);

    double norm = 0;
    for (int i = 0; i < size; i++)
    {
        res[i] = 0;
        for (int k = 0; k < size; k++)
        {
            res[i] += matrix[i][k] * x[k];
        }
        res[i] -= b[i];
        norm += pow(res[i], 2);
    }
    norm = sqrt(norm);
    std::cout << "Faktoryzacja LU - norma: " << norm << endl;

    for (int i = 0; i < size; i++)
    {
        delete[] low[i];
        delete[] up[i];
    }
    delete[] low;
    delete[] up;
    delete[] x;
    delete[] y;
    delete[] res;
}



void jacobiMethod(double** matrix, double* b, int size, bool showOutput)
{
    //init
    double** low = new double* [size];
    double** up = new double* [size];
    double* diagonalReversed = new double[size];
    double* r = new double[size];
    double* res = new double[size];
    double* temp = new double[size];

    for (int i = 0; i < size; i++)
    {
        low[i] = new double[size];
        up[i] = new double[size];
        r[i] = 1.0;
        for (int j = 0; j < size; j++)
        {
            if (i == j)
            {
                diagonalReversed[i] = 1 / matrix[i][i];
            }
            if (i > j)
                low[i][j] = matrix[i][j];
            else
                low[i][j] = 0.0;

            if (i < j)
                up[i][j] = matrix[i][j];
            else
                up[i][j] = 0.0;
        }
    }

    //jacobi
    double norm;
    int counter = 0;
    while (counter < 1000)
    {
        counter++;
        for (int i = 0; i < size; i++)
        {
            temp[i] = 0.0;
            for (int j = 0; j < size; j++)
            {
                temp[i] += (low[i][j] + up[i][j]) * r[j];
            }
        }

        for (int i = 0; i < size; i++)
            r[i] = -1 * diagonalReversed[i] * temp[i] + (diagonalReversed[i] * b[i]);

        norm = 0;
        for (int i = 0; i < size; i++)
        {
            res[i] = 0;
            for (int k = 0; k < size; k++)
            {
                res[i] += matrix[i][k] * r[k];
            }
            res[i] -= b[i];
            norm += pow(res[i], 2);
        }
        norm = sqrt(norm);
        if (showOutput)
            std::cout << "Iteracja: " << counter << " Norma: " << norm << endl;
        if (norm < residualError)
            break;
    }
    std::cout << "Liczba iteracji: " << counter << " Norma: " << norm << endl;
    for (int i = 0; i < size; i++)
    {
        delete[] low[i];
        delete[] up[i];
    }
    delete[] low;
    delete[] up;
    delete[] r;
    delete[] diagonalReversed;
    delete[] res;
    delete[] temp;
}

void gaussSeidlMethod(double** matrix, double* b, int size, bool showOutput)
{
    //init
    double** diagonalAndLow = new double* [size];
    double** up = new double* [size];
    double* Ur = new double[size];
    double* DLb = new double[size];
    double* DLUr = new double[size];
    double* r = new double[size];
    double* res = new double[size];
    for (int i = 0; i < size; i++)
    {
        diagonalAndLow[i] = new double[size];
        up[i] = new double[size];
        r[i] = 1.0;
        for (int j = 0; j < size; j++)
        {
            if (i == j or i > j)
            {
                diagonalAndLow[i][j] = matrix[i][j];
            }
            else
                diagonalAndLow[i][j] = 0.0;

            if (i < j)
                up[i][j] = matrix[i][j];
            else
                up[i][j] = 0.0;
        }
    }

    //gauss Seidl
    double norm;
    int counter = 0;
    while (counter < 1000)
    {
        counter++;

        for (int i = 0; i < size; i++)
        {
            Ur[i] = 0;
            DLb[i] = 0;
            DLUr[i] = 0;
            for (int j = 0; j < size; j++)
            {
                Ur[i] += -1 * up[i][j] * r[j];
            }
        }

        forwardSubstitution(DLUr, diagonalAndLow, Ur, size);

        forwardSubstitution(DLb, diagonalAndLow, b, size);

        for (int i = 0; i < size; i++)
        {
            r[i] = DLUr[i] + DLb[i];
        }

        norm = 0;
        for (int i = 0; i < size; i++)
        {
            res[i] = 0;
            for (int k = 0; k < size; k++)
            {
                res[i] += matrix[i][k] * r[k];
            }
            res[i] -= b[i];
            norm += pow(res[i], 2);
        }
        norm = sqrt(norm);
        if (showOutput)
            std::cout << "Iteracja: " << counter << " Norma: " << norm << endl;
        if (norm < residualError)
            break;
    }
    std::cout << "Liczba iteracji: " << counter << " Norma: " << norm << endl;
    for (int i = 0; i < size; i++)
    {
        delete[] diagonalAndLow[i];
        delete[] up[i];
    }
    delete[] up;
    delete[] r;
    delete[] diagonalAndLow;
    delete[] res;
}


int main()
{
    int size = N;
    int a1 = 11;
    int a2 = -1;
    int a3 = -1;
    double** matrix = new double* [size];
    double* b = new double[size];


    //fill matrix and vector with data with data
    for (int i = 0; i < size; i++)
    {
        matrix[i] = new double[size];
        b[i] = sin(i * (1 + f));
        for (int j = 0; j < size; j++)
        {
            if (i == j)
            {
                matrix[i][j] = a1;
            }
            else if (i + 1 == j || i - 1 == j)
                matrix[i][j] = a2;
            else if (i + 2 == j || i - 2 == j)
                matrix[i][j] = a3;
            else
                matrix[i][j] = 0.0;
        }
    }


    cout << "Zadanie B" << endl;
    cout << "Test metod dla \n N = " << size << "\n a1 = " << a1 << "\n a2 = " << a2 << "\n a3 = " << a3 << endl;

    cout << endl << "Metoda iteracyjna Jacobiego:" << endl;
    auto begin = chrono::high_resolution_clock::now();
    jacobiMethod(matrix, b, size, true);
    auto end = chrono::high_resolution_clock::now();
    auto elapsed = chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    std::cout << "Czas metoda iteracyjna Jacobiego: " << elapsed.count() * 1e-9 << "s" << endl;

    cout << endl << "Metoda iteracyjna Gaussa-Seidla:" << endl;
    begin = chrono::high_resolution_clock::now();
    gaussSeidlMethod(matrix, b, size, true);
    end = chrono::high_resolution_clock::now();
    elapsed = chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    std::cout << "Czas metoda iteracyjna Gaussa-Seidla: " << elapsed.count() * 1e-9 << "s" << endl;

    cout << endl << "Metoda bezposrednia - faktoryzacja LU:" << endl;
    begin = chrono::high_resolution_clock::now();
    factorizationLU(matrix, b, size);
    end = chrono::high_resolution_clock::now();
    elapsed = chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    std::cout << "Czas metoda bezpośrednia - faktoryzacja LU: " << elapsed.count() * 1e-9 << "s" << endl;

    a1 = 3;
    a2 = -1;
    a3 = -1;

    for (int i = 0; i < size; i++)
    {
        matrix[i] = new double[size];
        b[i] = sin(i * (1 + f));
        for (int j = 0; j < size; j++)
        {
            if (i == j)
            {
                matrix[i][j] = a1;
            }
            else if (i + 1 == j || i - 1 == j)
                matrix[i][j] = a2;
            else if (i + 2 == j || i - 2 == j)
                matrix[i][j] = a3;
            else
                matrix[i][j] = 0.0;
        }
    }

    cout << endl << "Zadanie C" << endl;
    cout << "Test metod dla \n N = " << size << "\n a1 = " << a1 << "\n a2 = " << a2 << "\n a3 = " << a3 << endl;

    cout << endl << "Metoda iteracyjna Jacobiego:" << endl;
    begin = chrono::high_resolution_clock::now();
    jacobiMethod(matrix, b, size, false);
    end = chrono::high_resolution_clock::now();
    elapsed = chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    std::cout << "Czas metoda iteracyjna Jacobiego: " << elapsed.count() * 1e-9 << "s" << endl;

    cout << endl << "Metoda iteracyjna Gaussa-Seidla:" << endl;
    begin = chrono::high_resolution_clock::now();
    gaussSeidlMethod(matrix, b, size, false);
    end = chrono::high_resolution_clock::now();
    elapsed = chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    std::cout << "Czas metoda iteracyjna Gaussa-Seidla: " << elapsed.count() * 1e-9 << "s" << endl;

    cout << endl << "Metoda bezposrednia - faktoryzacja LU:" << endl;
    begin = chrono::high_resolution_clock::now();
    factorizationLU(matrix, b, size);
    end = chrono::high_resolution_clock::now();
    elapsed = chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
    std::cout << "Czas metoda bezpośrednia - faktoryzacja LU: " << elapsed.count() * 1e-9 << "s" << endl;

    for (int i = 0; i < size; i++)
    {
        delete[] matrix[i];
    }
    delete[] matrix;
    delete[] b;

    int sizeArray[6] = { 100, 500, 750, 1000, 2000, 3000 };

    cout << endl << endl << "Zadanie E" << endl;


    a1 = 11;
    a2 = -1;
    a3 = -1;
    iterationData jacobiData[6];
    iterationData gaussData[6];
    iterationData factorizationData[6];
    
    for (int k = 0; k < 6; k++)
    {
        cout << endl << "Rozmiar macierzy:" << sizeArray[k] << endl;
        matrix = new double* [sizeArray[k]];
        b = new double[sizeArray[k]];
        for (int i = 0; i < sizeArray[k]; i++)
        {

            matrix[i] = new double[sizeArray[k]];
            b[i] = sin(i * (1 + f));
            for (int j = 0; j < sizeArray[k]; j++)
            {
                if (i == j)
                {
                    matrix[i][j] = a1;
                }
                else if (i + 1 == j || i - 1 == j)
                    matrix[i][j] = a2;
                else if (i + 2 == j || i - 2 == j)
                    matrix[i][j] = a3;
                else
                    matrix[i][j] = 0.0;
            }
        }

        begin = chrono::high_resolution_clock::now();
        jacobiMethod(matrix, b, sizeArray[k], false);
        end = chrono::high_resolution_clock::now();
        elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        std::cout << "Czas metoda iteracyjna Jacobiego: " << elapsed.count() * 1e-9 << "s" << endl;
        jacobiData[k].matrixSize = sizeArray[k];
        jacobiData[k].time = elapsed.count() * 1e-9;


        begin = chrono::high_resolution_clock::now();
        gaussSeidlMethod(matrix, b, sizeArray[k], false);
        end = chrono::high_resolution_clock::now();
        elapsed = chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        std::cout << "Czas metoda iteracyjna Gaussa-Seidla: " << elapsed.count() * 1e-9 << "s" << endl;
        gaussData[k].matrixSize = sizeArray[k];
        gaussData[k].time = elapsed.count() * 1e-9;

        begin = chrono::high_resolution_clock::now();
        factorizationLU(matrix, b, sizeArray[k]);
        end = chrono::high_resolution_clock::now();
        elapsed = chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        std::cout << "Czas metoda bezpośrednia - faktoryzacja LU: " << elapsed.count() * 1e-9 << "s" << endl;
        factorizationData[k].matrixSize = sizeArray[k];
        factorizationData[k].time = elapsed.count() * 1e-9;

        for (int i = 0; i < sizeArray[k]; i++)
        {
            delete[] matrix[i];
        }
        delete[] matrix;
        delete[] b;
    }

    ofstream fileOutput;
    fileOutput.open("gaussData.csv");
    fileOutput << "Size,Time(s)\n";
    for (int i = 0; i < 6; i++)
    {
        fileOutput << gaussData[i].matrixSize << "," << gaussData[i].time << "\n";
    }
    fileOutput.close();

    fileOutput.open("jacobiData.csv");
    fileOutput << "Size,Time(s)\n";
    for (int i = 0; i < 6; i++)
    {
        fileOutput << jacobiData[i].matrixSize << "," << jacobiData[i].time << "\n";
    }
    fileOutput.close();

    fileOutput.open("factorizationData.csv");
    fileOutput << "Size,Time(s)\n";
    for (int i = 0; i < 6; i++)
    {
        fileOutput << factorizationData[i].matrixSize << "," << factorizationData[i].time << "\n";
    }
    fileOutput.close();
}