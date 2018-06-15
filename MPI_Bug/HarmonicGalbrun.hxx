#ifndef MONTJOIE_FILE_HARMONIC_GALBRUN_HXX

namespace Montjoie
{

  template<class Dimension>
  class VarGalbrunIndex_Base;

  template<class TypeEquation>
  class VarGalbrun_Eq;
  
  //! class to specify resolution of harmonic Galbrun equation for any flow
  /*!
    Galbrun's equation is written as 
    \rho (-i omega + sigma + M \cdot \nabla)^2 u - \nabla( \rho c^2 div u) + (div u) \nabla p_0 - (\nabla u)^T \nabla p_0 = f
    where (\nabla u)^T = | du_x/dx du_y/dx |
                         | du_x/dy du_y/dy |
                             
    rho, c, M, p_0 and sigma are given coefficients (stationary flow)
    Two intermediary unknowns p and v are introduced to obtain a first-order formulation
    \rho (-i omega + sigma + M \cdot \nabla) u - rho v = 0
    \rho (-i omega + sigma + M \cdot \nabla) v - \nabla( rho c^2 p)
          + p \nabla p_0  -  (\nabla u)^T \nabla p_0 = f
    p - div u = 0
    
    Dirichlet condition : u.n = 0
    Neumann condition p = 0 (i.e. div u = 0)
   */
  template<class T, class Dim>
  class GalbrunEquation_Base : public GenericEquation<T, 1>
  {
  public :
    typedef Dim Dimension;

    static const bool FirstOrderFormulation = true;
    
    enum {nb_unknowns = 1+2*Dimension::dim_N, nb_unknowns_hdg=0,
	  nb_components_en = 1, nb_components_hn = 1,
	  nb_unknowns_scal = Dimension::dim_N, nb_unknowns_vec = Dimension::dim_N+1};
    
    static inline bool SymmetricGlobalMatrix() { return false; }
    static inline bool SymmetricElementaryMatrix() { return false; }

    static void SetIndexToCompute(VarGalbrunIndex_Base<Dimension>& var);

    // for compatbility purpose
    template<class TypeEquation>
    static void ComputeMassMatrix(EllipticProblem<TypeEquation>& var,
                                  int i, const ElementReference<Dimension, 1>&);
        
    // For a detailed description of the following methods
    // look at the file GenericEquation.hxx (class GenericEquation_Base)
    template<class TypeEquation, class T0, class Vector1>
    static void GetNeededDerivative(const EllipticProblem<TypeEquation>& vars,
                                    const GlobalGenericMatrix<T0>& nat_mat,
				    Vector1& unknown_to_derive, Vector1& fct_test_to_derive);
    
    template<class TypeEquation, class T0, class MatStiff>
    static void GetGradPhiTensor(const EllipticProblem<TypeEquation>& vars,
                                 int num_elem, int jloc,
				 const GlobalGenericMatrix<T0>& nat_mat, int ref,
                                 MatStiff& Dgrad_phi, MatStiff& Ephi_grad);
    
    template<class TypeEquation, class T0, class Vector1, class Vector2>
    static void ApplyGradientUnknown(const EllipticProblem<TypeEquation>& var,
				     int i, int j, const GlobalGenericMatrix<T0>& nat_mat,
                                     int ref, Vector1& dU, Vector2& V);

    template<class TypeEquation, class T0, class Vector1, class Vector2>
    static void ApplyGradientFctTest(const EllipticProblem<TypeEquation>& var,
				     int i, int j, const GlobalGenericMatrix<T0>& nat_mat,
                                     int ref, Vector1& U, Vector2& dV);
    
    template<class TypeEquation, class T0, class MatMass>
    static void GetTensorMass(const EllipticProblem<TypeEquation>& vars,
			      int i, int j, const GlobalGenericMatrix<T0>& nat_mat, int ref, MatMass& mass);
    
    template<class TypeEquation, class T0, class Vector1>
    static void ApplyTensorMass(const EllipticProblem<TypeEquation>& var,
                                int i, int j, const GlobalGenericMatrix<T0>& nat_mat,
				int ref, Vector1& U, Vector1& V);
    
    template<class Matrix1, class GenericPb, class T0>
    static void GetNabc(Matrix1& Nabc, const typename Dimension::R_N& normale,
			int ref, int iquad, int k, const GlobalGenericMatrix<T0>& nat_mat, int ref_d, 
			const GenericPb& vars, const ElementReference<Dimension, 1>& Fb);
    
    template<class Vector1, class TypeEquation, class T0>
    static void MltNabc(typename Dimension::R_N& normale, int ref, const Vector1& Vn,
                        Vector1& Un, int num_elem1,
			int num_point, const GlobalGenericMatrix<T0>& nat_mat, int ref_d, 
			const EllipticProblem<TypeEquation>& vars, const ElementReference<Dimension, 1>& );
    
    template<class Matrix1, class GenericPb, class T0>
    static void GetPenalDG(Matrix1& Nabc, typename Dimension::R_N& normale, int iquad, int k,
			   int nf, const GlobalGenericMatrix<T0>& nat_mat, int ref, int ref2,
                           const GenericPb& vars, const ElementReference<Dimension, 1>& Fb);
    
    template<class Vector1, class Vector2, class GenericPb, class T0>
    static void MltPenalDG(const typename Dimension::R_N& normale, const Vector1& Vn, Vector2& Un,
			   int i, int k, int nf, const GlobalGenericMatrix<T0>& nat_mat,
                           int ref, int ref2, const GenericPb& vars, const ElementReference<Dimension, 1>& Fb);
    
  };
  
  
  //! stationary Galbrun equation for any flow
  /*!
    The equations are the same as in GalbrunEquation_Base where
    -i omega is replaced by the mass coefficient and other terms are multiplied by stiffness coefficient
   */
  template<class Dimension>
  class GalbrunStationaryEquation : public GalbrunEquation_Base<Real_wp, Dimension>
  {
  public :
  };

  
  //! time-harmonic Galbrun equation for any flow
  /*!
    The equations are given in description of GalbrunEquation_Base
   */
  template<class Dimension>
  class HarmonicGalbrunEquation : public GalbrunEquation_Base<Complex_wp, Dimension>
  {
  public :
  };

  
  //! class to specify resolution of harmonic Galbrun equation for any flow
  /*!
    Galbrun's equation is written as 
    \rho (-i omega + sigma + M \cdot \nabla)^2 u - \nabla( \rho c^2 div u) + (div u) \nabla p_0 - (\nabla u)^T \nabla p_0 = f

    rho, c, M, p_0 and sigma are given coefficients (stationary flow)    
    Two intermediary unknowns q and p are introduced to obtain a first-order formulation
    rho (-i omega + sigma + M \cdot \nabla) u - \nabla p - rho q = 0
    rho (-i omega + sigma + M \cdot \nabla) q - \nabla sigma p - (\nabla M)^T \nabla p 
                 - M \cdot \nabla rho / rho \nabla p + (div u) \nabla p_0 - (\nabla u)^T \nabla p_0 = f
    rho (-i omega + sigma + M \cdot \nabla) p - (rho c)^2 div u = 0

    Dirichlet condition : u.n = 0
    Neumann condition div u = 0 (p= 0)
   */
  template<class T, class Dim>
  class GalbrunEquationDG_Base : public GenericEquation<T, 1>
  {
  public :
    typedef Dim Dimension;
    
    static const bool FirstOrderFormulation = true;
    
    enum {nb_unknowns = 1+2*Dimension::dim_N, nb_unknowns_hdg=1,
	  nb_components_en = 1, nb_components_hn = 1,	  
	  nb_unknowns_scal = Dimension::dim_N, nb_unknowns_vec = Dimension::dim_N+1};

    static inline bool SymmetricGlobalMatrix() { return false; }
    static inline bool SymmetricElementaryMatrix() { return false; }

    static void SetIndexToCompute(VarGalbrunIndex_Base<Dimension>& var);
    
    // for compatbility purpose
    template<class TypeEquation>
    static void ComputeMassMatrix(EllipticProblem<TypeEquation>& var,
                                  int num_elem, const ElementReference<Dimension, 1>& Fb);
        
    // For a detailed description of the following methods
    // look at the file GenericEquation.hxx (class GenericEquation_Base)
    template<class GenericPb, class T0, class Vector1>
    static void GetNeededDerivative(const GenericPb& vars, const GlobalGenericMatrix<T0>& nat_mat,
				    Vector1& unknown_to_derive, Vector1& fct_test_to_derive);
    
    template<class TypeEquation, class T0, class MatStiff>
    static void GetGradPhiTensor(const EllipticProblem<TypeEquation>& vars,
                                 int num_elem, int jloc, const GlobalGenericMatrix<T0>& nat_mat, int ref,
                                 MatStiff& Dgrad_phi, MatStiff& Ephi_grad);
    
    template<class TypeEquation, class T0, class Vector1, class Vector2>
    static void ApplyGradientUnknown(const EllipticProblem<TypeEquation>& vars,
				     int iquad, int k, const GlobalGenericMatrix<T0>& nat_mat,
                                     int ref, Vector1& Vn, Vector2& Un);
    
    template<class TypeEquation, class T0, class MatMass>
    static void GetTensorMass(const EllipticProblem<TypeEquation>& vars,
			      int iquad, int k, const GlobalGenericMatrix<T0>& nat_mat, int ref, MatMass& Cj);
    
    template<class TypeEquation, class T0, class Vector1>
    static void ApplyTensorMass(const EllipticProblem<TypeEquation>& var,
                                int iquad, int k, const GlobalGenericMatrix<T0>& nat_mat,
				int ref, Vector1& Un, Vector1& Vn);
    
    template<class T0>
    static void GetAbsoluteD(TinyMatrix<T0, General, 5, 5>& Dtest, const R2& normale,
			     const Real_wp& c0, bool non_unif, const Real_wp&);

    template<class T0>
    static void GetAbsoluteD(TinyMatrix<T0, General, 7, 7>& Dtest, const R3& normale,
			     const Real_wp& c0, bool non_unif, const Real_wp&);
    
    template<class T0, class GenericPb>
    static void GetAbsoluteMatrixD(TinyMatrix<T0, General, 5, 5>& D, const R2& normale, int iquad, int k,
                                   const GlobalGenericMatrix<T0>& nat_mat, const GenericPb& vars);

    template<class T0, class GenericPb>
    static void GetAbsoluteMatrixD(TinyMatrix<T0, General, 7, 7>& D, const R3& normale, int iquad, int k,
                                   const GlobalGenericMatrix<T0>& nat_mat, const GenericPb& vars);
    
    template<class Matrix1, class GenericPb, class T0>
    static void GetNabc(Matrix1& Nabc, const typename Dimension::R_N& normale,
			int ref, int iquad, int npoint,
			const GlobalGenericMatrix<T0>& nat_mat, int ref2, 
			const GenericPb& vars, const ElementReference<Dimension, 1>& Fb);
    
    template<class Vector1, class TypeEquation, class T0>
    static void MltNabc(typename Dimension::R_N& normale, int ref,
                        const Vector1& Vn, Vector1& Un, int num_elem1,
                        int num_point, const GlobalGenericMatrix<T0>& nat_mat, int ref2, 
                        const EllipticProblem<TypeEquation>& vars, const ElementReference<Dimension, 1>&);
    
    template<class Matrix1, class GenericPb, class T0>
    static void GetPenalDG(Matrix1& Nabc, typename Dimension::R_N& normale, int iquad, int npoint,
			   int nf, const GlobalGenericMatrix<T0>& nat_mat, int ref, int ref2,
                           const GenericPb& vars, const ElementReference<Dimension, 1>& Fb);
    
    template<class Vector1, class Vector2, class GenericPb, class T0>
    static void MltPenalDG(const typename Dimension::R_N& normale, const Vector1& Vn, Vector2& Un,
                           int iquad, int npoint, int nf, const GlobalGenericMatrix<T0>& nat_mat,
                           int ref, int ref2, const GenericPb& vars,
			   const ElementReference<Dimension, 1>& Fb);
    
  };
  
  
  //! stationary Galbrun's equation for any flow
  template<class Dimension>
  class GalbrunStationaryEquationDG : public GalbrunEquationDG_Base<Real_wp, Dimension>
  {
  public :
  };

  
  //! time-harmonic Galbrun's equation for any flow
  template<class Dimension>
  class HarmonicGalbrunEquationDG : public GalbrunEquationDG_Base<Complex_wp, Dimension>
  {
  public :
  };


  //! Base class handling indexes in aeroacoustics (Galbrun or Linearized Euler)
  template<class Dimension>
  class VarGalbrunIndex_Base
  {
    typedef typename Dimension::R_N R_N;
    typedef typename Dimension::MatrixN_N MatrixN_N;
    
  public:
    //! damping
    Vector<ScalarPhysicalIndice<Dimension, Real_wp> > ref_sigma;
    //! density rho
    Vector<ScalarPhysicalIndice<Dimension, Real_wp> > ref_rho0;
    //! sound speed and pressure
    Vector<ScalarPhysicalIndice<Dimension, Real_wp> > ref_c0, ref_p0;
    //!  value of the variable flow M
    Vector<VectorPhysicalIndice<Dimension, Dimension::dim_N, Real_wp> > ref_v0 ; //ref_f0;

    //! evaluation of M, grad(rho), grad(sigma), grad(p_0) and grad(c_0) on quadrature points
    Vector<Vector<R_N> > eval_flow, grad_rho0, grad_sigma, grad_p0, grad_c0, grad_gamma;
    Vector<Vector<R_N> > eval_f0; //AJOUT NATHAN -- Source f0 (momentum)
    
    //! evaluation of rho, rho c^2, c, sigma and div(M) on quadrature points
    Vector<VectReal_wp> eval_rho0, eval_rhoC2, eval_c0, eval_sigma, div_flow, eval_gamma;
    
    //! gradient of flow : grad(M)
    Vector<Vector<MatrixN_N> > grad_flow;
    Vector<Vector<MatrixN_N> > grad_f0; //AJOUT NATHAN -- Grad source f_0
    
    //! hessian of pressure
    Vector<Vector<TinyMatrix<Real_wp, Symmetric, Dimension::dim_N, Dimension::dim_N> > > hessian_p0;
    
    //! hessian of flow
    Vector<Vector<TinyVector<TinyMatrix<Real_wp, Symmetric, Dimension::dim_N, Dimension::dim_N>,
			     Dimension::dim_N> > > hessian_flow;
    
    //! true if flow is adjusted to check Neumann condition
    bool adjustment_neumann;
    
    bool compute_grad_flow, compute_div_flow, compute_grad_rho, compute_grad_sigma;
    bool compute_grad_c0, compute_hessian_flow, compute_hessian_p0;
    bool compute_gamma, store_grad_rho0_c0, compute_grad_gamma;
    bool compute_source_momentum, compute_grad_momentum; //AJOUT NATHAN
    
  private:
    DistributedProblem<Dimension>& var_problem;
    
  public:
    template<class TypeEquation>
    VarGalbrunIndex_Base(EllipticProblem<TypeEquation>& var);

    // additional parameters of the data file
    void SetInputData(const string& description_field, const VectString& parameters);
    
    void InitIndices(int n);
    int GetNbPhysicalIndices() const;
    void SetIndices(int i, const VectString& parameters);
    void SetPhysicalIndex(const string& name_media, int i, const VectString& parameters);
    string GetPhysicalIndexName(int m) const;
    bool IsVaryingMedia(int i) const;
    Real_wp GetVelocityOfMedia(int ref) const;
    Real_wp GetVelocityOfInfinity() const;
    
    void GetVaryingIndices(Vector<PhysicalVaryingMedia<Dimension, Real_wp>* >& rho_real,
			   IVect& num_ref, IVect& num_index, IVect& num_component,
			   Vector<bool>& compute_grad, Vector<bool>& compute_hess);
    
    void ComputePhysicalCoefficients();
    
  };
  
  //! stationary or time-harmonic base class for Galbrun's equation
  /*!
    Galbrun's equation is written as 
    \rho (-i omega + sigma + M \cdot \nabla)^2 u - \nabla( \rho c^2 div u) + (div u) \nabla p_0 - (\nabla u)^T \nabla p_0 = f
    where (\nabla u)^T = | du_x/dx du_y/dx |
                         | du_x/dy du_y/dy |

    rho, c, M, p_0 and sigma are given coefficients (stationary flow)
   */
  class VarGalbrun_Base
  {
  public :
    bool apply_convective_derivate_source;
    int drop_unstable_terms;
    Real_wp coef_convective_term;
    enum { DROP_NONE, DROP_CONVECTIVE, DROP_NON_UNIFORM};

  protected:
    VarProblem_Base& var_problem;
    
  public:
    template<class TypeEquation>
    VarGalbrun_Base(EllipticProblem<TypeEquation>& var);
    
    // additional parameters of the data file
    void SetInputData(const string& description_field, const VectString& parameters);
        
    void CheckBoundaryCondition(const IVect& boundary_condition);
    
  };

  
  template<class Complexe, class Dimension>
  class VarGalbrun_Dim : public VarGalbrun_Base, public VarGalbrunIndex_Base<Dimension>
  {
  public:
    template<class TypeEquation>
    VarGalbrun_Dim(EllipticProblem<TypeEquation>& var);

    void SetInputData(const string& description_field, const VectString& parameters);
    
    template<class T0>
    void ModifyVolumetricSource(int i, int j, const typename Dimension::R_N& x,
				const VirtualSourceField<T0, Dimension>& fsrc,
				Vector<T0>& f) const;
  };
  

  //! class used to solve harmonic Galbrun equation with DG method
  template<class Dimension>
  class EllipticProblem<HarmonicGalbrunEquation<Dimension> >
    : public VarGalbrun_Eq<HarmonicGalbrunEquation<Dimension> >
  {
  public:
    void AddElementaryFluxesDG(VirtualMatrix<Real_wp>& mat_sp,
			       const GlobalGenericMatrix<Real_wp>& nat_mat,
			       int offset_row = 0, int offset_col = 0);

    void AddElementaryFluxesDG(VirtualMatrix<Complex_wp>& mat_sp,
			       const GlobalGenericMatrix<Complex_wp>& nat_mat,
			       int offset_row = 0, int offset_col = 0);

    void ComputeElementaryMatrix(int, IVect&, Matrix<Real_wp>&,
				 CondensationBlockSolver_Base<Real_wp>&,
				 const GlobalGenericMatrix<Real_wp>&);
    
    void ComputeElementaryMatrix(int, IVect&, Matrix<Complex_wp>&,
				 CondensationBlockSolver_Base<Complex_wp>&,
				 const GlobalGenericMatrix<Complex_wp>&);

  };


  //! class used to solve real Galbrun equation with DG method
  template<class Dimension>
  class EllipticProblem<GalbrunStationaryEquation<Dimension> >
    : public VarGalbrun_Eq<GalbrunStationaryEquation<Dimension> >
  {
  public:
    
  };


  //! class used to solve harmonic Galbrun equation with DG method
  template<class Dimension>
  class EllipticProblem<HarmonicGalbrunEquationDG<Dimension> >
    : public VarGalbrun_Eq<HarmonicGalbrunEquationDG<Dimension> >
  {
  public:
    void AddElementaryFluxesDG(VirtualMatrix<Real_wp>& mat_sp,
			       const GlobalGenericMatrix<Real_wp>& nat_mat,
			       int offset_row = 0, int offset_col = 0);

    void AddElementaryFluxesDG(VirtualMatrix<Complex_wp>& mat_sp,
			       const GlobalGenericMatrix<Complex_wp>& nat_mat,
			       int offset_row = 0, int offset_col = 0);

    void ComputeElementaryMatrix(int, IVect&, Matrix<Real_wp>&,
				 CondensationBlockSolver_Base<Real_wp>&,
				 const GlobalGenericMatrix<Real_wp>&);
    
    void ComputeElementaryMatrix(int, IVect&, Matrix<Complex_wp>&,
				 CondensationBlockSolver_Base<Complex_wp>&,
				 const GlobalGenericMatrix<Complex_wp>&);
    
  };


  //! class used to solve real Galbrun equation with DG method
  template<class Dimension>
  class EllipticProblem<GalbrunStationaryEquationDG<Dimension> >
    : public VarGalbrun_Eq<GalbrunStationaryEquationDG<Dimension> >
  {
  public:
    
  };

  template<class Dimension>
  class FemMatrixFreeClass<Real_wp, GalbrunStationaryEquationDG<Dimension> >
    : public FemMatrixFreeClass_Eq<Real_wp, GalbrunStationaryEquationDG<Dimension> >
  {
  public:
    
    FemMatrixFreeClass(const EllipticProblem<GalbrunStationaryEquationDG<Dimension> >& var_);
    
    virtual void MltAddFree(const GlobalGenericMatrix<Real_wp>& nat_mat,
			    const SeldonTranspose&, int lvl, 
			    const Vector<Real_wp>& X, Vector<Real_wp>& Y) const;

    virtual void MltAddFree(const GlobalGenericMatrix<Real_wp>& nat_mat,
			    const SeldonTranspose&, int lvl, 
			    const Vector<Complex_wp>& X, Vector<Complex_wp>& Y) const;
    
  };
  
}

#define MONTJOIE_FILE_HARMONIC_GALBRUN_HXX
#endif
