#include <vector>
#include <string>
#include <complex>
#include <exception>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "dirft.h"
#include "finufft.h"

namespace py = pybind11;

// The FFTW options
enum FFTWOptions {
  estimate=FFTW_ESTIMATE,
  measure=FFTW_MEASURE,
  patient=FFTW_PATIENT,
  exhaustive=FFTW_EXHAUSTIVE
};

// A custom error handler to propagate errors back to Python
class finufft_error : public std::exception {
public:
  finufft_error (const std::string& msg) : msg_(msg) {};
  virtual const char* what() const throw() {
    return msg_.c_str();
  }
private:
  std::string msg_;
};

// A few macros to help reduce copying


// The interface functions:
#define ASSEMBLE_OPTIONS               \
  nufft_opts opts;                     \
  opts.R = R;                          \
  opts.debug = debug;                  \
  opts.spread_debug = spread_debug;    \
  opts.spread_sort = spread_sort;      \
  opts.fftw = fftw;

#define CHECK_FLAG(NAME)                            \
  if (ier != 0) {                                   \
    std::ostringstream msg;                         \
    msg << #NAME << " failed with code " << ier;    \
    throw finufft_error(msg.str());                 \
  }

// ------------
// 1D INTERFACE
// ------------

py::array_t<CPX> nufft1d1(
  py::array_t<FLT> xj, py::array_t<CPX> cj,
  INT ms,
  FLT eps, int iflag, FLT R, int debug, int spread_debug, int spread_sort, FFTWOptions fftw
) {
  // Check the dimensions
  auto buf_x = xj.request(),
       buf_c = cj.request();
  if (buf_x.ndim != 1 || buf_c.ndim != 1)
    throw finufft_error("xj and cj must be 1-dimensional");
  if (buf_x.size != buf_c.size)
    throw finufft_error("xj and cj must be the same length");
  long n = buf_x.size;

  ASSEMBLE_OPTIONS

  // Allocate output
  auto result = py::array_t<CPX>(ms);
  auto buf_F = result.request();

  // Run the driver
  int ier = finufft1d1(
    n, (FLT*)buf_x.ptr, (CPX*)buf_c.ptr,
    iflag, eps, ms, (CPX*)buf_F.ptr,
    opts
  );

  CHECK_FLAG(nufft1d1)
  return result;
}


py::array_t<CPX> nufft1d2(
  py::array_t<FLT> xj, py::array_t<CPX> fk,
  FLT eps, int iflag, FLT R, int debug, int spread_debug, int spread_sort, FFTWOptions fftw
) {
  // Check the dimensions
  auto buf_x = xj.request(),
       buf_F = fk.request();
  if (buf_x.ndim != 1 || buf_F.ndim != 1)
    throw finufft_error("xj and fk must be 1-dimensional");
  long n = buf_x.size, ms = buf_F.size;

  ASSEMBLE_OPTIONS

  // Allocate output
  auto result = py::array_t<CPX>(n);
  auto buf_c = result.request();

  // Run the driver
  int ier = finufft1d2(
    n, (FLT*)buf_x.ptr, (CPX*)buf_c.ptr,
    iflag, eps, ms, (CPX*)buf_F.ptr,
    opts
  );

  CHECK_FLAG(nufft1d2)
  return result;
}

py::array_t<CPX> nufft1d3(
  py::array_t<FLT> xj, py::array_t<CPX> cj,
  py::array_t<FLT> s,
  FLT eps, int iflag, FLT R, int debug, int spread_debug, int spread_sort, FFTWOptions fftw
) {
  // Check the dimensions
  auto buf_x = xj.request(),
       buf_c = cj.request();
  if (buf_x.ndim != 1 || buf_c.ndim != 1)
    throw finufft_error("xj and cj must be 1-dimensional");
  if (buf_x.size != buf_c.size)
    throw finufft_error("xj and cj must be the same length");
  long n = buf_x.size;

  auto buf_s = s.request();
  if (buf_s.ndim != 1)
    throw finufft_error("s must be 1-dimensional");
  long nk = buf_s.size;

  ASSEMBLE_OPTIONS

  // Allocate output
  auto result = py::array_t<CPX>(nk);
  auto buf_F = result.request();

  // Run the driver
  int ier = finufft1d3(
    n, (FLT*)buf_x.ptr, (CPX*)buf_c.ptr,
    iflag, eps, nk, (FLT*)buf_s.ptr, (CPX*)buf_F.ptr,
    opts
  );

  CHECK_FLAG(nufft1d3)
  return result;
}

// ------------
// 2D INTERFACE
// ------------

py::array_t<CPX> nufft2d1(
  py::array_t<FLT> xj, py::array_t<FLT> yj, py::array_t<CPX> cj,
  INT ms, INT mt,
  FLT eps, int iflag, FLT R, int debug, int spread_debug, int spread_sort, FFTWOptions fftw
) {
  // Check the dimensions
  auto buf_x = xj.request(),
       buf_y = yj.request(),
       buf_c = cj.request();
  if (buf_x.ndim != 1 || buf_y.ndim != 1 || buf_c.ndim != 1)
    throw finufft_error("xj, yj, and cj must be 1-dimensional");
  if (buf_x.size != buf_y.size || buf_x.size != buf_c.size)
    throw finufft_error("xj, yj, and cj must be the same length");
  long n = buf_x.size;

  ASSEMBLE_OPTIONS

  // Allocate output
  auto result = py::array_t<CPX>({size_t(mt), size_t(ms)});
  auto buf_F = result.request();

  // Run the driver
  int ier = finufft2d1(
    n, (FLT*)buf_x.ptr, (FLT*)buf_y.ptr, (CPX*)buf_c.ptr,
    iflag, eps, ms, mt, (CPX*)buf_F.ptr,
    opts
  );

  CHECK_FLAG(nufft2d1)
  return result;
}

py::array_t<CPX> nufft2d2(
  py::array_t<FLT> xj, py::array_t<FLT> yj, py::array_t<CPX> fk,
  FLT eps, int iflag, FLT R, int debug, int spread_debug, int spread_sort, FFTWOptions fftw
) {
  // Check the dimensions
  auto buf_x = xj.request(),
       buf_y = yj.request(),
       buf_F = fk.request();
  if (buf_x.ndim != 1 || buf_y.ndim != 1)
    throw finufft_error("xj and yj must be 1-dimensional; fk must be 2-dimensional");
  long n = buf_x.size, ms = buf_F.shape[1], mt = buf_F.shape[0];

  ASSEMBLE_OPTIONS

  // Allocate output
  auto result = py::array_t<CPX>(n);
  auto buf_c = result.request();

  // Run the driver
  int ier = finufft2d2(
    n, (FLT*)buf_x.ptr, (FLT*)buf_y.ptr, (CPX*)buf_c.ptr,
    iflag, eps, ms, mt, (CPX*)buf_F.ptr,
    opts
  );

  CHECK_FLAG(nufft2d2)
  return result;
}

py::array_t<CPX> nufft2d3(
  py::array_t<FLT> xj, py::array_t<FLT> yj, py::array_t<CPX> cj,
  py::array_t<FLT> s, py::array_t<FLT> t,
  FLT eps, int iflag, FLT R, int debug, int spread_debug, int spread_sort, FFTWOptions fftw
) {
  // Check the dimensions
  auto buf_x = xj.request(),
       buf_y = yj.request(),
       buf_c = cj.request();
  if (buf_x.ndim != 1 || buf_y.ndim != 1 || buf_c.ndim != 1)
    throw finufft_error("xj, yj, and cj must be 1-dimensional");
  if (buf_x.size != buf_y.size || buf_x.size != buf_c.size)
    throw finufft_error("xj, yj, and cj must be the same length");
  long n = buf_x.size;

  auto buf_s = s.request(),
       buf_t = t.request();
  if (buf_s.ndim != 1 || buf_t.ndim != 1)
    throw finufft_error("s and t must be 1-dimensional");
  if (buf_s.size != buf_t.size)
    throw finufft_error("s and t must be the same length");
  long nk = buf_s.size;

  ASSEMBLE_OPTIONS

  // Allocate output
  auto result = py::array_t<CPX>(nk);
  auto buf_F = result.request();

  // Run the driver
  int ier = finufft2d3(
    n, (FLT*)buf_x.ptr, (FLT*)buf_y.ptr, (CPX*)buf_c.ptr,
    iflag, eps, nk, (FLT*)buf_s.ptr, (FLT*)buf_t.ptr, (CPX*)buf_F.ptr,
    opts
  );

  CHECK_FLAG(nufft2d3)
  return result;
}

// ----------------
// DIRECT INTERFACE
// ----------------

py::array_t<CPX> dirft1d1_(
  py::array_t<FLT> xj, py::array_t<CPX> cj, INT ms, int iflag
) {
  // Check the dimensions
  auto buf_x = xj.request(),
       buf_c = cj.request();
  if (buf_x.ndim != 1 || buf_c.ndim != 1)
    throw finufft_error("xj and cj must be 1-dimensional");
  if (buf_x.size != buf_c.size)
    throw finufft_error("xj and cj must be the same length");
  long n = buf_x.size;

  // Allocate output
  auto result = py::array_t<CPX>(ms);
  auto buf_F = result.request();

  // Run the driver
  dirft1d1(
    n, (FLT*)buf_x.ptr, (CPX*)buf_c.ptr,
    iflag, ms, (CPX*)buf_F.ptr
  );

  return result;
}

py::array_t<CPX> dirft1d2_(
  py::array_t<FLT> xj, py::array_t<CPX> fk, int iflag
) {
  // Check the dimensions
  auto buf_x = xj.request(),
       buf_F = fk.request();
  if (buf_x.ndim != 1 || buf_F.ndim != 1)
    throw finufft_error("xj and fk must be 1-dimensional");
  long n = buf_x.size,
       ms = buf_F.size;

  // Allocate output
  auto result = py::array_t<CPX>(n);
  auto buf_c = result.request();

  // Run the driver
  dirft1d2(
    n, (FLT*)buf_x.ptr, (CPX*)buf_c.ptr,
    iflag, ms, (CPX*)buf_F.ptr
  );

  return result;
}

py::array_t<CPX> dirft1d3_(
  py::array_t<FLT> xj, py::array_t<CPX> cj, py::array_t<FLT> s, int iflag
) {
  // Check the dimensions
  auto buf_x = xj.request(),
       buf_c = cj.request(),
       buf_s = s.request();
  if (buf_x.ndim != 1 || buf_c.ndim != 1 || buf_s.ndim != 1)
    throw finufft_error("xj, cj, and s must be 1-dimensional");
  if (buf_x.size != buf_c.size)
    throw finufft_error("xj and cj must be the same length");
  long n = buf_x.size,
       ms = buf_s.size;

  // Allocate output
  auto result = py::array_t<CPX>(ms);
  auto buf_F = result.request();

  // Run the driver
  dirft1d3(
    n, (FLT*)buf_x.ptr, (CPX*)buf_c.ptr,
    iflag, ms, (FLT*)buf_s.ptr, (CPX*)buf_F.ptr
  );

  return result;
}

py::array_t<CPX> dirft2d1_(
  py::array_t<FLT> xj, py::array_t<FLT> yj, py::array_t<CPX> cj, INT ms, INT mt, int iflag
) {
  // Check the dimensions
  auto buf_x = xj.request(),
       buf_y = yj.request(),
       buf_c = cj.request();
  if (buf_x.ndim != 1 || buf_y.ndim != 1 || buf_c.ndim != 1)
    throw finufft_error("xj, yj, and cj must be 1-dimensional");
  if (buf_x.size != buf_y.size || buf_x.size != buf_c.size)
    throw finufft_error("xj, yj, and cj must be the same length");
  long n = buf_x.size;

  // Allocate output
  auto result = py::array_t<CPX>({size_t(mt), size_t(ms)});
  auto buf_F = result.request();

  // Run the driver
  dirft2d1(
    n, (FLT*)buf_x.ptr, (FLT*)buf_y.ptr, (CPX*)buf_c.ptr,
    iflag, ms, mt, (CPX*)buf_F.ptr
  );

  return result;
}

py::array_t<CPX> dirft2d2_(
  py::array_t<FLT> xj, py::array_t<FLT> yj, py::array_t<CPX> fk, int iflag
) {
  // Check the dimensions
  auto buf_x = xj.request(),
       buf_y = yj.request(),
       buf_F = fk.request();
  if (buf_x.ndim != 1 || buf_y.ndim != 1 || buf_F.ndim != 2)
    throw finufft_error("xj and yj must be 1-dimensional; fk must be 2-dimensional");
  long n = buf_x.size, ms = buf_F.shape[1], mt = buf_F.shape[0];

  // Allocate output
  auto result = py::array_t<CPX>(n);
  auto buf_c = result.request();

  // Run the driver
  dirft2d2(
    n, (FLT*)buf_x.ptr, (FLT*)buf_y.ptr, (CPX*)buf_c.ptr,
    iflag, ms, mt, (CPX*)buf_F.ptr
  );

  return result;
}

py::array_t<CPX> dirft2d3_(
  py::array_t<FLT> xj, py::array_t<FLT> yj, py::array_t<CPX> cj,
  py::array_t<FLT> s, py::array_t<FLT> t, int iflag
) {
  // Check the dimensions
  auto buf_x = xj.request(),
       buf_y = yj.request(),
       buf_c = cj.request();
  if (buf_x.ndim != 1 || buf_y.ndim != 1 || buf_c.ndim != 1)
    throw finufft_error("xj, yj, and cj must be 1-dimensional");
  if (buf_x.size != buf_y.size || buf_x.size != buf_c.size)
    throw finufft_error("xj, yj, and cj must be the same length");
  long n = buf_x.size;

  auto buf_s = s.request(),
       buf_t = t.request();
  if (buf_s.ndim != 1 || buf_t.ndim != 1)
    throw finufft_error("s and t must be 1-dimensional");
  if (buf_s.size != buf_t.size)
    throw finufft_error("s and t must be the same length");
  long nk = buf_s.size;

  // Allocate output
  auto result = py::array_t<CPX>(nk);
  auto buf_F = result.request();

  // Run the driver
  dirft2d3(
    n, (FLT*)buf_x.ptr, (FLT*)buf_y.ptr, (CPX*)buf_c.ptr,
    iflag, nk, (FLT*)buf_s.ptr, (FLT*)buf_t.ptr, (CPX*)buf_F.ptr
  );

  return result;
}


PYBIND11_PLUGIN(interface) {
  py::module m("interface", R"delim(
Docs
)delim");

  // Deal with custom exceptions
  py::register_exception<finufft_error>(m, "FINUFFTError");

  // Export the FFTW options
  py::enum_<FFTWOptions>(m, "FFTWOptions")
      .value("estimate", FFTWOptions::estimate)
      .value("measure", FFTWOptions::measure)
      .value("patient", FFTWOptions::patient)
      .value("exhaustive", FFTWOptions::exhaustive)
      .export_values();

  // ------------
  // 1D INTERFACE
  // ------------

  m.def("nufft1d1", &nufft1d1, R"delim(
Type-1 1D complex nonuniform FFT

::

              nj-1
     fk(k1) = SUM cj[j] exp(+/-i k1 xj(j))  for -ms/2 <= k1 <= (ms-1)/2
              j=0

Args:
    xj (float[n]): location of sources on interval [-pi, pi]
    cj (complex[n]): FLT complex array of source strengths
    ms (int): number of Fourier modes computed, may be even or odd;
        in either case the mode range is integers lying in [-ms/2, (ms-1)/2]
    eps (float): precision requested (>1e-16)
    iflag (int): if >=0, uses + sign in exponential, otherwise - sign.

Returns:
    fk (complex[ms]): FLT complex array of Fourier transform values

)delim",
    py::arg("xj"),
    py::arg("cj"),
    py::arg("ms"),
    py::arg("eps") = 1.0e-9,
    py::arg("iflag") = 1,
    py::arg("R") = 2.0,
    py::arg("debug") = 0,
    py::arg("spread_debug") = 0,
    py::arg("spread_sort") = 1,
    py::arg("fftw") = FFTWOptions::estimate
  );

  m.def("nufft1d2", &nufft1d2, R"delim(
Type-2 1D complex nonuniform FFT

::

     cj[j] = SUM   fk[k1] exp(+/-i k1 xj[j])      for j = 0,...,nj-1
             k1
     where sum is over -ms/2 <= k1 <= (ms-1)/2.

Args:
    xj (float[n]): location of sources on interval [-pi, pi]
    fk (complex[ms]): complex FLT array of nj answers at targets
    eps (float): precision requested (>1e-16)
    iflag (int): if >=0, uses + sign in exponential, otherwise - sign.

Returns:
    cj (complex[n]): FLT complex array of source strengths

)delim",
    py::arg("xj"),
    py::arg("fk"),
    py::arg("eps") = 1.0e-9,
    py::arg("iflag") = 1,
    py::arg("R") = 2.0,
    py::arg("debug") = 0,
    py::arg("spread_debug") = 0,
    py::arg("spread_sort") = 1,
    py::arg("fftw") = FFTWOptions::estimate
  );

  m.def("nufft1d3", &nufft1d3, R"delim(
Type-3 1D complex nonuniform FFT.

::

               nj-1
     fk[k]  =  SUM   c[j] exp(+-i s[k] xj[j]),      for k = 0, ..., nk-1
               j=0

Args:
    xj (float[n]): location of sources in R
    cj (complex[n]): FLT complex array of source strengths
    s (float[nk]): frequency locations of targets in R
    eps (float): precision requested (>1e-16)
    iflag (int): if >=0, uses + sign in exponential, otherwise - sign.

Returns:
     fk (complex[nk]): complex FLT array of nk answers at targets

)delim",
    py::arg("xj"),
    py::arg("cj"),
    py::arg("s"),
    py::arg("eps") = 1.0e-9,
    py::arg("iflag") = 1,
    py::arg("R") = 2.0,
    py::arg("debug") = 0,
    py::arg("spread_debug") = 0,
    py::arg("spread_sort") = 1,
    py::arg("fftw") = FFTWOptions::estimate
  );

  // ------------
  // 2D INTERFACE
  // ------------

  m.def("nufft2d1", &nufft2d1, R"delim(
Type-1 2D complex nonuniform FFT.

::

                  nj-1
     f[k1,k2] =   SUM  c[j] exp(+-i (k1 x[j] + k2 y[j]))
                  j=0

     for -ms/2 <= k1 <= (ms-1)/2,  -mt/2 <= k2 <= (mt-1)/2.

Args:
    xj (float[n]): location of sources on interval [-pi, pi]
    yj (float[n]): location of sources on interval [-pi, pi]
    cj (complex[n]): FLT complex array of source strengths
    ms (int): number of Fourier modes computed, may be even or odd;
        in either case the mode range is integers lying in [-ms/2, (ms-1)/2]
    mt (int): number of Fourier modes computed, may be even or odd;
        in either case the mode range is integers lying in [-ms/2, (ms-1)/2]
    eps (float): precision requested (>1e-16)
    iflag (int): if >=0, uses + sign in exponential, otherwise - sign.

Returns:
    fk (complex[mt, ms]): FLT complex array of Fourier transform values

)delim",
    py::arg("xj"),
    py::arg("yj"),
    py::arg("cj"),
    py::arg("ms"),
    py::arg("mt"),
    py::arg("eps") = 1.0e-9,
    py::arg("iflag") = 1,
    py::arg("R") = 2.0,
    py::arg("debug") = 0,
    py::arg("spread_debug") = 0,
    py::arg("spread_sort") = 1,
    py::arg("fftw") = FFTWOptions::estimate
  );

  m.def("nufft2d2", &nufft2d2, R"delim(
Type-2 2D complex nonuniform FFT

::

    cj[j] =  SUM   fk[k1,k2] exp(+/-i (k1 xj[j] + k2 yj[j]))
            k1,k2
    for j = 0,...,nj-1
    where sum is over -ms/2 <= k1 <= (ms-1)/2, -mt/2 <= k2 <= (mt-1)/2,

Args:
    xj (float[n]): location of sources on interval [-pi, pi]
    yj (float[n]): location of sources on interval [-pi, pi]
    fk (complex[mt, ms]): complex FLT array of nj answers at targets
    eps (float): precision requested (>1e-16)
    iflag (int): if >=0, uses + sign in exponential, otherwise - sign.

Returns:
    cj (complex[n]): FLT complex array of source strengths

)delim",
    py::arg("xj"),
    py::arg("yj"),
    py::arg("fk"),
    py::arg("eps") = 1.0e-9,
    py::arg("iflag") = 1,
    py::arg("R") = 2.0,
    py::arg("debug") = 0,
    py::arg("spread_debug") = 0,
    py::arg("spread_sort") = 1,
    py::arg("fftw") = FFTWOptions::estimate
  );

  m.def("nufft2d3", &nufft2d3, R"delim(
Type-3 2D complex nonuniform FFT.

::

               nj-1
     fk[k]  =  SUM   c[j] exp(+-i (s[k] xj[j] + t[k] yj[j]),  for k=0,...,nk-1
               j=0

Args:
    xj (float[n]): location of sources in R
    yj (float[n]): location of sources in R
    cj (complex[n]): FLT complex array of source strengths
    s (float[nk]): frequency locations of targets in R.
    t (float[nk]): frequency locations of targets in R.
    eps (float): precision requested (>1e-16)
    iflag (int): if >=0, uses + sign in exponential, otherwise - sign.

Returns:
     fk (complex[nk]): complex FLT array of nk answers at targets

)delim",
    py::arg("xj"),
    py::arg("yj"),
    py::arg("cj"),
    py::arg("s"),
    py::arg("t"),
    py::arg("eps") = 1.0e-9,
    py::arg("iflag") = 1,
    py::arg("R") = 2.0,
    py::arg("debug") = 0,
    py::arg("spread_debug") = 0,
    py::arg("spread_sort") = 1,
    py::arg("fftw") = FFTWOptions::estimate
  );

  // DIRECT
  m.def("dirft1d1", &dirft1d1_, "Type-1 1D direct",
    py::arg("xj"),
    py::arg("cj"),
    py::arg("ms"),
    py::arg("iflag") = 1
  );
  m.def("dirft1d2", &dirft1d2_, "Type-2 1D direct",
    py::arg("xj"),
    py::arg("fk"),
    py::arg("iflag") = 1
  );
  m.def("dirft1d3", &dirft1d3_, "Type-3 1D direct",
    py::arg("xj"),
    py::arg("fk"),
    py::arg("s"),
    py::arg("iflag") = 1
  );

  m.def("dirft2d1", &dirft2d1_, "Type-1 2D direct",
    py::arg("xj"),
    py::arg("yj"),
    py::arg("cj"),
    py::arg("ms"),
    py::arg("mt"),
    py::arg("iflag") = 1
  );

  m.def("dirft2d2", &dirft2d2_, "Type-2 2D direct",
    py::arg("xj"),
    py::arg("yj"),
    py::arg("fk"),
    py::arg("iflag") = 1
  );

  m.def("dirft2d3", &dirft2d3_, "Type-3 2D direct",
    py::arg("xj"),
    py::arg("yj"),
    py::arg("cj"),
    py::arg("s"),
    py::arg("t"),
    py::arg("iflag") = 1
  );

  return m.ptr();
}
