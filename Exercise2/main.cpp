#include"Eigen/Eigen"
#include<math.h>
#include<iostream>
#include <iomanip>
#include <Eigen/Dense>


using namespace Eigen;
using namespace std;

VectorXd PALU(const MatrixXd A, const VectorXd b) {
    PartialPivLU<MatrixXd> lu(A);
    VectorXd x = lu.solve(b);
    return x;
}

VectorXd QR(const MatrixXd& A, const VectorXd& b) {
    HouseholderQR<MatrixXd> qr(A);
    VectorXd x = qr.solve(b);
    return x;
}

double errRel(const VectorXd& sol, const VectorXd& x) {
    return (sol - x).norm() / sol.norm();
}
int main()
{
    //scrivo le 3 matrici con i rispettivi termini noti e la soluzione (uguale per tutti i sistemi)
    Eigen::Vector2d sol;
    sol<<-1.0e+0, -1.0e+00;

    Eigen::Matrix2d A;
    A<<5.547001962252291e-01, -3.770900990025203e-02,
        8.320502943378437e-01, -9.992887623566787e-01;
    Eigen::Vector2d a;
    a<< -5.169911863249772e-01, 1.672384680188350e-01;

    cout << "Sistema Ax=a:\n";
    PALU(A, a);
    QR(A, a);
    cout << "Errore relativo con fattorizzazione PALU: " << errRel(sol, PALU(A, a)) << endl;
    cout << "Errore relativo con fattorizzazione QR: " << errRel(sol, QR(A, a)) << endl;


    Eigen::Matrix2d B;
    B<<5.547001962252291e-01, -5.540607316466765e-01,
        8.320502943378437e-01, -8.324762492991313e-01;
    Eigen::Vector2d b;
    b<<-6.394645785530173e-04, 4.259549612877223e-04;
    cout << "Sistema Bx=b:\n";
    PALU(B, b);
    QR(B, b);
    cout << "Errore relativo con fattorizzazione PALU: " << errRel(sol, PALU(B,b)) << endl;
    cout << "Errore relativo con fattorizzazione QR: " << errRel(sol, QR(B, b)) << endl;


    Eigen::Matrix2d C;
    C<<5.547001962252291e-01, -5.547001955851905e-01,
        8.320502943378437e-01, -8.320502947645361e-01;
    Eigen::Vector2d c;
    c<<-6.400391328043042e-10, 4.266924591433963e-10;
    cout << "Sistema Cx=c:\n";
    PALU(C, c);
    QR(C, c);
    cout << "Errore relativo con fattorizzazione PALU: " << errRel(sol, PALU(C, c)) << endl;
    cout << "Errore relativo con fattorizzazione QR: " << errRel(sol, QR(C, c)) << endl;
    return 0;
}
