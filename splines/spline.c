#include <stdbool.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stddef.h>
#include <stdio.h>
#include "spline.h"

void kToolMath_matrixMult1d2d(size_t const X, size_t const Y, float const A[X], float const B[X][Y], float out[Y]) {
	for (size_t i = 0; i < Y; ++i)
		for (size_t j = 0; j < X; ++j)
			out[i] += A[j] * B[j][i];
}

void kToolMath_matrixInvert(size_t const X, float const in[X][X], float out[X][X]) {
	// augment matrix with identity
	size_t const COLS = 2 * X;
        float b[X][COLS];
        for (size_t i = 0; i < X; ++i)
                for (size_t j = 0; j < X; ++j) {
                        b[i][j] = in[i][j];
                        b[i][j+X] = (i == j);
                }

	// Gaussian Elimination with Partial Pivoting
        for (size_t i = 0; i < X; ++i) {
		// pivoting (row swapping)
		for (size_t j = i+1; j < X; ++j)
			if (fabsf(b[j][i]) > fabsf(b[i][i]))
				for (size_t k = 0; k < COLS; ++k) {
					float const temp = b[i][k];
					b[i][k] = b[j][k];
					b[j][k] = temp;
				}

                float const div = 1.0f / b[i][i];
		for (size_t j = 0; j < COLS; ++j)
                        b[i][j] *= div;

                for (size_t m = 0; m < X; ++m) {
                        if (m == i)
                                continue;

                        float const x = b[m][i];
                        for (size_t j = 0; j < COLS; ++j)
                                b[m][j] -= x * b[i][j];
                }
        }

	// extract inverse matrix
        for (size_t i = 0; i < X; ++i)
                for (size_t j = 0; j < X; ++j)
                        out[i][j] = b[i][j+X];
}

/*
  takes:
  n (number of cer points)
  cer.t, cer.psin, cer.rot, cer.ti, cer.ti_error
  paramdata[BA_FITTING_PSINLIM-1]->cerDxMin
  targets[FSTA_FITTING_CSMOOTH-1]

  returns:
  mHat (associated with the input psin)
*/

int calculate_mhat(size_t n,
		   float psin[n], float rot[n],
		   float dxMin, float p,
		   float mPsin[n], float mHat[n]) {

  struct CerSort {
    float psin;
    float rot;
  } cerSort[n];
  for (size_t i = 0; i < n; ++i) {
    cerSort[i].psin = psin[i];
    cerSort[i].rot = rot[i];
  }

  int compare(void const * A, void const * B) {
    struct CerSort const * a = A;
    struct CerSort const * b = B;
    return (a->psin > b->psin) - (a->psin < b->psin);
  }
  qsort(&cerSort, n, sizeof(struct CerSort), compare);

  size_t newN = 0;
  for (size_t i = 1; i < n; ++i) {
    struct CerSort * const a = &cerSort[newN];
    struct CerSort * const b = &cerSort[i];
    if (fabsf(b->psin - a->psin) < dxMin) {
      a->psin = (a->psin + b->psin) * 0.5f;
      a->rot = (a->rot + b->rot) * 0.5f;
    } else {
      ++newN;
      cerSort[newN] = *b;
    }
  }
  n = newN;

  float h[n-1];
  float dy[n-1];
  for (size_t i = 0; i < n-1; ++i) {
    h[i] = cerSort[i+1].psin - cerSort[i].psin;
    dy[i] = cerSort[i+1].rot - cerSort[i].rot;
  }

  float uncert[n];
  for (size_t i = 0; i < n; ++i)
    uncert[i] = fmaxf(10.0f, fabsf(0.2f * cerSort[i].rot));
  float delta[n-2][n];
  memset(delta, 0, sizeof(delta));
  for (size_t i = 0; i < n-2; ++i) {
    float const u0 = sqrtf(uncert[i + 0]);
    float const u1 = sqrtf(uncert[i + 1]);
    float const u2 = sqrtf(uncert[i + 2]);
    float const hInv0 = 1.0f / h[i + 0];
    float const hInv1 = 1.0f / h[i + 1];
    delta[i][i + 0] = hInv0 / u0;
    delta[i][i + 1] = -hInv0 / u1 - hInv1 / u1;
    delta[i][i + 2] = hInv1 / u2;
  }

  float W[n-2][n-2];
  memset(W, 0, sizeof(W));
  for (size_t i = 0; i < n-2; ++i)
    W[i][i] = (h[i] + h[i + 1]) * 2.0f;
  for (size_t i = 1; i < n-2; ++i) {
    W[i][i-1] = h[i];
    W[i-1][i] = h[i];
  }

  float A[n-2][n-2];
  memset(A, 0, sizeof(A));
  for (size_t i = 0; i < n-2; ++i)
    for (size_t j = 0; j < n; ++j)
      for (size_t k = 0; k < n-2; ++k)
	//delta[k][j] is really deltaTransverse[j][k] without the intermediate storage
	A[i][k] += delta[i][j] * delta[k][j];
  //float const p = targets[FSTA_FITTING_CSMOOTH-1];
  for (size_t i = 0; i < n-2; ++i)
    for (size_t j = 0; j < n-2; ++j) {
      A[i][j] *= 6.0f * (1.0f - p);
      A[i][j] += p * W[i][j];
    }
  float AInv[n-2][n-2];
  kToolMath_matrixInvert(n-2, A, AInv);

  float b[n-2];
  for (size_t i = 0; i < n-2; ++i)
    b[i] = dy[i + 1] / h[i + 1] - dy[i] / h[i];

  float u[n-2];
  memset(u, 0, sizeof(u));
  kToolMath_matrixMult1d2d(n-2, n-2, b, AInv, u);

  float d2u[n];
  d2u[0] = u[0] / h[0];
  d2u[1] = (u[1] - u[0]) / h[1] - d2u[0];
  for (size_t i = 2; i < n-2; ++i)
    d2u[i] = (u[i] - u[i-1]) / h[i] - (u[i-1] - u[i-2]) / h[i-1];
  d2u[n-2] = -u[n-3] / h[n-2] - (u[n-3] - u[n-4]) / h[n-3];
  d2u[n-1] = u[n-3] / h[n-2];

  for (size_t i = 0; i < n; ++i) {
    mPsin[i] = cerSort[i].psin;
    mHat[i] = cerSort[i].rot - 6.0f * (1.0f - p) * d2u[i] / uncert[i];
  }
  return 0;
}

int spline_fit_natural(struct spline * spl, spline_t * scratch) {
	if (scratch == NULL)
		return 3;

	size_t const n = spl->n;
	spline_t * subdiag = scratch;
	spline_t * diag = scratch + n;
	spline_t * superdiag = diag + n;

	// 1st and nth equations from boundary conditions
	diag[0] = 1.0;
	superdiag[0] = 0.0;
	diag[n-1] = 1.0;
	subdiag[n-2] = -1.0;

	// diagonals
	for (size_t i = 1; i < n-1; ++i) {
		diag[i] = 2.0 * (spl->x[i+1] - spl->x[i-1]);
		superdiag[i]= spl->x[i+1] - spl->x[i];
		subdiag[i-1] = spl->x[i] - spl->x[i-1];
	}

	spl->c[0] = 0;
	spl->c[n-1] = 0;
	for (size_t i = 1; i < n-1; ++i)
		spl->c[i] = 3.0 * (
			((spl->y[i+1] - spl->y[i])   / (spl->x[i+1] - spl->x[i])) -
			((spl->y[i]   - spl->y[i-1]) / (spl->x[i]   - spl->x[i-1])));

	/*
	 * O(n) Tridiagonal system solver: Thomas algorithm.
	 * Note: not guaranteed to be stable and destroys original input. 
	 * Reference: http://www.industrial-maths.com/ms6021_thomas.pdf
	 * x -- input vector, function returns solution
	 * n -- number of equations
	 * a -- subdiagonal
	 * b -- main diagonal
	 * c -- superdiagonal
	 */
	void trilus(size_t n, spline_t x[n], spline_t a[n], spline_t b[n], spline_t c[n]) {
		// Forward sweep
		for (size_t i = 1; i < n; ++i) {
			spline_t const m = a[i-1] / b[i-1];
			b[i] = b[i] - (m * c[i-1]);
			x[i] = x[i] - (m * x[i-1]);
		}

		x[n-1] /= b[n-1];

		// Backwards sweep
		for(size_t i = n-1; i > 0; --i)
			x[i-1] = (x[i-1] - c[i-1] * x[i]) / b[i-1];
	}

	trilus(spl->n, spl->c, subdiag, diag, superdiag);

	return 0;
}

int spline_eval(size_t N, spline_t values[N], spline_t const eval_pts[N], struct spline spl, spline_t scratch[3 * spl.n]) {
	if (values == NULL)
		return 1;
	else if (eval_pts == NULL)
		return 2;

	if (scratch)
		spline_fit_natural(&spl, scratch);

	// Binary search for index
	size_t const M = spl.n;
	spline_t const * X = spl.x;
	spline_t const * Y = spl.y;
	spline_t const * C = spl.c;
	for (size_t i = 0; i < N; ++i) {
		size_t idx = 0;
		size_t high = M - 1;
		size_t low = 0;
		size_t mid = low + ((high - low) / 2);
		if (eval_pts[i] <= X[0])
			idx = 0;
		else if (eval_pts[i] >= X[M-1])
			idx = M - 2;
		else
			while (low < high) {
				mid = low + ((high - low) / 2);
				if (X[mid] <= eval_pts[i]) {
					idx = mid + 1;
					low = mid + 1;
				} else if (X[mid] > eval_pts[i]) {
					high = mid;
				}
			}

		if (idx > 0)
			--idx;
		else if (idx > M - 2)
			idx = M - 2;

		spline_t const b_i =
			((Y[idx+1] - Y[idx]) / (X[idx+1] - X[idx])) -
			(((X[idx+1] - X[idx]) * (C[idx+1] + (2.0 * C[idx]))) / 3.0);
		spline_t const d_i = (C[idx+1] - C[idx]) / (3.0 * (X[idx+1] - X[idx]));
		values[i] = Y[idx] +
			b_i    * (eval_pts[i] - X[idx]) +
			C[idx] * (eval_pts[i] - X[idx]) * (eval_pts[i] - X[idx]) +
			d_i    * (eval_pts[i] - X[idx]) * (eval_pts[i] - X[idx]) * (eval_pts[i] - X[idx]);
	}
}

#define NFIT 121
#define N 4
int main(void) {
  float rot[N]={0.1,0.3,0.5,0.9};
  float psin[N]={0.1,0.2,0.5,0.9};
  printf("Input rotation:\n");
  for (int i=0; i<N; i++)
    printf("%f\t",rot[i]);
  printf("\n");
  float mPsin[N]={0};
  float mHat[N]={0};
  float const dxMin=0.01;
  int const p=0.5;
  calculate_mhat(N, psin, rot, dxMin, p, mPsin, mHat);
  printf("psin:\n");
  for (int i=0; i<N; i++)
    printf("%f\t",mPsin[i]);
  printf("\n");
  printf("mhat:\n");
  for (int i=0; i<N; i++)
    printf("%f\t",mHat[i]);
  printf("\n");

  spline_t psiNSpline[N];
  spline_t mHatSpline[N];
  for (int i=0; i<N; i++) {
    psiNSpline[i] = (spline_t) mPsin[i];
    mHatSpline[i] = (spline_t) mHat[i];
  }
  spline_t eval[NFIT];
  for (size_t i = 0; i < NFIT; ++i)
    eval[i] = i * 0.01f;
  spline_t work[N];
  spline_t dummy[3*N];
  spline_t v[NFIT];
  struct spline s = { .n = N, .x = psiNSpline, .y = mHatSpline, .c = work };
  spline_eval(NFIT, v, eval, s, dummy);

  printf("Splined output:\n");
  for (int i=0; i<NFIT; i++)
    printf("%f\t",v[i]);
  printf("\n");
}
