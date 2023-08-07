#include <stddef.h>

// Real spline type
typedef double spline_t;

struct spline {
	size_t n;
	spline_t * x;
	spline_t * y;
	spline_t * c;
};

void pcs_debug(size_t N, spline_t psiNSpline[N], spline_t mHatSpline[N]);

/* Given spline with values y on grid x, fits a natural spline and stores fitted second derivative coefficients in c.
 * NOTE: scratch should be at least 3*sizeof(spline_t)*n, where n is the number of interpolation points.
 *
 * Return values:
 * 	0: success
 * 	1: values is NULL
 * 	2: eval_pts is NULL
 * 	3: scratch is NULL
 */
int spline_fit_natural(struct spline * spl, spline_t * scratch);

/* Evalutes spline at numEval given points. Stores result in values.  If isFit flag is set, function will fit the
 * spline before evaluating.  As before, client is responsible for allocating enough scratch space:
 *     3 * sizeof(spline_t) * n
 * if fitting is desired. If only evaluation is deisred, a null pointer can be passed. */
int spline_eval(size_t n, spline_t values[n], spline_t const eval_pts[n], struct spline spl, spline_t scratch[3 * spl.n]);

