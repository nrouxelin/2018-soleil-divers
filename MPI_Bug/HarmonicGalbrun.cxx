#ifndef MONTJOIE_FILE_HARMONIC_GALBRUN_CXX

namespace Montjoie
{

  /************************
   * GalbrunEquation_Base *
   ************************/


  template<class T, class Dimension>
    void GalbrunEquation_Base<T, Dimension>
    ::SetIndexToCompute(VarGalbrunIndex_Base<Dimension>& var)
    {
      var.compute_div_flow = false;
      var.compute_grad_flow = true; //AJOUT NATHAN (normalement false)
      var.compute_grad_rho = true;
      var.compute_grad_sigma = false;
      var.compute_grad_c0 = false;
      var.compute_hessian_flow = true;
      var.compute_hessian_p0 = true;
      var.compute_gamma = false;
      var.compute_grad_gamma = false;
      var.compute_source_momentum = true; //AJOUT NATHAN
      var.compute_grad_momentum = false; //AJOUT NATHAN
    }


  //! for compatbility purpose
  template<class T, class Dimension> template<class TypeEquation>
    void GalbrunEquation_Base<T, Dimension>::
    ComputeMassMatrix(EllipticProblem<TypeEquation>& var,
        int num_elem, const ElementReference<Dimension, 1>& Fb)
    {
    }


  //! which derivatives to evaluate during matrix-vector product ?
  template<class T, class Dimension> template<class TypeEquation, class T0, class Vector1>
    void GalbrunEquation_Base<T, Dimension>::
    GetNeededDerivative(const EllipticProblem<TypeEquation>& vars,
        const GlobalGenericMatrix<T0>& nat_mat,
        Vector1& unknown_to_derive, Vector1& fct_test_to_derive)
    {
      fct_test_to_derive.Fill(true);
      unknown_to_derive.Fill(true);
    }


  //! fills tensors D and E appearing in the variational formulation
  /*!
    \param[in] vars considered problem
    \param[in] num_elem element where D and E must be computed
    \param[in] jloc quadrature point where D and E must be computed
    \param[in] nat_mat object containing mass and stiffness coefficients
    \param[in] ref reference of the element
    \param[out] Ephi_grad tensor E
    \param[out] Dphi_grad tensor D    
    The tensors D and E are appearing in the terms
    \int_K E \nabla u v + D u \nabla v dx
    of the variational formulation
    */  
  template<class T, class Dimension>
    template<class TypeEquation, class T0, class MatStiff>
    void GalbrunEquation_Base<T, Dimension>::
    GetGradPhiTensor(const EllipticProblem<TypeEquation>& vars,
        int num_elem, int jloc,
        const GlobalGenericMatrix<T0>& nat_mat, int ref,
        MatStiff& Dgrad_phi, MatStiff& Ephi_grad)
    {
      FillZero(Dgrad_phi);
      FillZero(Ephi_grad);

      TinyVector<Real_wp, Dimension::dim_N> v0 = vars.eval_flow(num_elem)(jloc);

      Real_wp rho = vars.eval_rho0(num_elem)(jloc);

      for (int i = 0; i < Dimension::dim_N; i++)
      {
        // acoustic part
        // term -grad(rho c^2 p) in equation of v
        Dgrad_phi(i+Dimension::dim_N, 2*Dimension::dim_N)(i) = vars.eval_rhoC2(num_elem)(jloc);
        // term -div u in equation of p
        Ephi_grad(2*Dimension::dim_N, i)(i) = -1.0;

        // flow part
        for (int p = 0; p < Dimension::dim_N; p++)
        {
          // term rho M \cdot \nabla u in equation of u
          Ephi_grad(p, p)(i) = rho*v0(i);	

          // term (div u) grad(p_0) in equation of v
          Ephi_grad(Dimension::dim_N+i, p)(p) += vars.grad_p0(num_elem)(jloc)(i);
          //AJOUT NATHAN
          if(vars.compute_source_momentum)
            Ephi_grad(Dimension::dim_N+i, p)(p)
              -=  vars.eval_f0(num_elem)(jloc)(i);

          // term - grad(u)^T grad(p_0) in equation of v
          Ephi_grad(i+Dimension::dim_N, p)(i) -= vars.grad_p0(num_elem)(jloc)(p);
        }

        // term rho M \cdot \nabla v in equation of v
        for (int p = Dimension::dim_N; p < 2*Dimension::dim_N; p++)
          Ephi_grad(p, p)(i) = rho*v0(i);	
      }

      Ephi_grad *= nat_mat.GetCoefStiffness();
      Dgrad_phi *= nat_mat.GetCoefStiffness();
    }


  //! Applying the tensor E to grad(v)
  /*!
    \param[in] var problem to be solved
    \param[in] i element number
    \param[in] j quadrature point number inside the element
    \param[in] nat_mat mass and stiffness coefficients
    \param[in] ref reference of the physical domain
    \param[in] Vn gradient of the unknown vector V
    \param[out] Un result to E grad(v)
    */
  template<class T, class Dimension>
    template<class TypeEquation, class T0, class Vector1, class Vector2>
    void GalbrunEquation_Base<T, Dimension>
    ::ApplyGradientUnknown(const EllipticProblem<TypeEquation>& vars,
        int num_elem, int jloc, const GlobalGenericMatrix<T0>& nat_mat,
        int ref, Vector1& Vn, Vector2& Un)
    {
      Un.Fill(0);
      TinyVector<Real_wp, Dimension::dim_N> v0 = vars.eval_flow(num_elem)(jloc);

      Real_wp rho = vars.eval_rho0(num_elem)(jloc);
      for (int j = 0; j < Dimension::dim_N; j++)
      {
        // acoustic part
        // term -div u in equation of p
        Un(2*Dimension::dim_N) -= Vn(j)(j);

        // flow part
        for (int p = 0; p < Dimension::dim_N; p++)
        {
          // term rho M \cdot \nabla u in equation of u
          Un(p) += rho*v0(j)*Vn(p)(j);

          // term (div u) grad(p_0) in equation of v
          Un(Dimension::dim_N+j) += vars.grad_p0(num_elem)(jloc)(j)*Vn(p)(p);
          // AJOUT NATHAN
          if(vars.compute_source_momentum)
            Un(Dimension::dim_N+j) -= vars.eval_f0(num_elem)(jloc)(j)*Vn(p)(p);

          // term - grad(u)^T grad(p_0) in equation of v
          Un(j+Dimension::dim_N) -= vars.grad_p0(num_elem)(jloc)(p)*Vn(p)(j);
        }

        // term rho M \cdot \nabla v in equation of u
        for (int p = Dimension::dim_N; p < 2*Dimension::dim_N; p++)
          Un(p) += rho*v0(j)*Vn(p)(j);
      }

      Mlt(nat_mat.GetCoefStiffness(), Un);
    }


  //! Applying the tensor D to u
  /*!
    \param[in] var problem to be solved
    \param[in] i element number
    \param[in] j quadrature point number inside the element
    \param[in] nat_mat mass and stiffness coefficients
    \param[in] ref reference of the physical domain
    \param[in] U unknown vector U
    \param[out] dV result D u
    */
  template<class T, class Dimension>
    template<class TypeEquation, class T0, class Vector1, class Vector2>
    void GalbrunEquation_Base<T, Dimension>
    ::ApplyGradientFctTest(const EllipticProblem<TypeEquation>& var,
        int num_elem, int jloc, const GlobalGenericMatrix<T0>& nat_mat,
        int ref, Vector1& U, Vector2& dV)
    {
      FillZero(dV);

      // term -grad(rho c^2 p)
      T0 coef = nat_mat.GetCoefStiffness();
      coef *= var.eval_rhoC2(num_elem)(jloc);
      for (int i = 0; i < Dimension::dim_N; i++)
        dV(i+Dimension::dim_N)(i) = coef*U(2*Dimension::dim_N);
    }


  //! Applies matrix A to a vector
  /*!
    \param[in] var given problem
    \param[in] i number of the element where M needs to be evaluated
    \param[in] j number of the local quadrature point in the element
    \param[in] nat_mat mass and stiffness coefficients
    \param[in] ref reference of the physical domain
    \param[in] U vector to be multiplied by A
    \param[out] V result vector V = A U
    */
  template<class T, class Dimension>
    template<class TypeEquation, class T0, class Vector1>
    void GalbrunEquation_Base<T, Dimension>::
    ApplyTensorMass(const EllipticProblem<TypeEquation>& var,
        int num_elem, int jloc, const GlobalGenericMatrix<T0>& nat_mat,
        int ref, Vector1& Un, Vector1& Vn)
    {
      Real_wp rho = var.eval_rho0(num_elem)(jloc);
      Real_wp sigma = var.eval_sigma(num_elem)(jloc);
      T m_iomega; var.GetMiomega(m_iomega);
      T0 coef 
        = rho*(m_iomega*nat_mat.GetCoefMass() + sigma*nat_mat.GetCoefDamping());

      // term sigma rho u and sigma rho v
      for (int i = 0; i < 2*Dimension::dim_N; i++)
        Vn(i) = coef*Un(i);

      for (int i = 0; i < Dimension::dim_N; i++)
      {
        // term -rho v in equation of u
        Vn(i) -= rho*Un(i+Dimension::dim_N);

        //AJOUT NATHAN -- term \nabla f in equation of v
        if(var.compute_grad_momentum){
          for(int l=0; l<Dimension::dim_N; l++){
            Vn(i+Dimension::dim_N) -= var.grad_f0(num_elem)(jloc)(i,l)*Un(i+Dimension::dim_N);
          }
        }
      }

      // term p in equation of p
      Vn(2*Dimension::dim_N) = Un(2*Dimension::dim_N);

      Mlt(nat_mat.GetCoefStiffness(), Vn);
    }


  //! returns the matrix A, in the integral \f$ \int_K A u v \f$ 
  /*!
    \param[in] vars given problem
    \param[in] i number of the element where A needs to be evaluated
    \param[in] j number of the local quadrature point in the element
    \param[in] nat_mat mass and stiffness coefficients
    \param[in] ref reference of the physical domain
    \param[out] mass matrix A
    */  
  template<class T, class Dimension>
    template<class TypeEquation, class T0, class MatMass>
    void GalbrunEquation_Base<T, Dimension>
    ::GetTensorMass(const EllipticProblem<TypeEquation>& vars,
        int num_elem, int jloc, const GlobalGenericMatrix<T0>& nat_mat, int ref, MatMass& Cj)
    {
      Cj.Fill(0);    
      // term rho (-i omega + sigma)
      Real_wp rho = vars.eval_rho0(num_elem)(jloc);    
      Real_wp sigma = vars.eval_sigma(num_elem)(jloc);
      T m_iomega; vars.GetMiomega(m_iomega);
      T0 coef 
        = rho*(m_iomega*nat_mat.GetCoefMass() + sigma*nat_mat.GetCoefDamping());

      for (int i = 0; i < 2*Dimension::dim_N; i++)
        Cj(i, i) = coef;

      for (int i = 0; i < Dimension::dim_N; i++)
      {
        // term -rho v in equation of u
        Cj(i, i+Dimension::dim_N) = -rho*nat_mat.GetCoefStiffness();

        //AJOUT NATHAN -- term \nabla f in equation of v
        if(vars.compute_grad_momentum){
          for(int l=0; l<Dimension::dim_N; l++)
            Cj(i+Dimension::dim_N,i+l+Dimension::dim_N) = -vars.grad_f0(num_elem)(jloc)(i,l);
        }
      }

      // term p in equation of p
      Cj(2*Dimension::dim_N, 2*Dimension::dim_N) = nat_mat.GetCoefStiffness();    

    }


  //! computation of matrix N associated to the boundary condition
  /*!
    \param[out] Nabc matrix N
    \param[in] normale outward normale
    \param[in] ref reference of the boundary
    \param[in] iquad element number
    \param[in] npoint local quadrature point number
    \param[in] nat_mat object containing mass and stiffness coefficients
    \param[in] ref2 reference of the element
    \param[in] vars considered problem
    \param[in] Fb finite element associated with the element
    */  
  template<class T, class Dimension>
    template<class Matrix1, class GenericPb, class T0>
    void GalbrunEquation_Base<T, Dimension>
    ::GetNabc(Matrix1& Nabc, const typename Dimension::R_N& normale,
        int ref, int iquad, int npoint, const GlobalGenericMatrix<T0>& nat_mat, int ref_d, 
        const GenericPb& vars, const ElementReference<Dimension, 1>& Fb)
    {
      int cond = vars.mesh.GetBoundaryCondition(ref);
      Nabc.Fill(0);

      //typename Dimension::R_N v0 = vars.eval_flow(iquad)(npoint);

      if (cond == BoundaryConditionEnum::LINE_DIRICHLET)
      {
        for (int i = 0; i < Dimension::dim_N; i++)
        {
          Nabc(i+Dimension::dim_N, 2*Dimension::dim_N)
            = -vars.eval_rhoC2(iquad)(npoint)*normale(i);

          Nabc(2*Dimension::dim_N, i) = normale(i);
        }

        Mlt(nat_mat.GetCoefStiffness(), Nabc);
      }
      else if (cond == BoundaryConditionEnum::LINE_NEUMANN)
      {
        for (int i = 0; i < Dimension::dim_N; i++)
        {
          Nabc(i+Dimension::dim_N, 2*Dimension::dim_N)
            = vars.eval_rhoC2(iquad)(npoint)*normale(i);

          Nabc(2*Dimension::dim_N, i) = -normale(i);
        }

        Mlt(nat_mat.GetCoefStiffness(), Nabc);
      }
      else if (cond == BoundaryConditionEnum::LINE_ABSORBING)
      {
        T0 coef;
        T m_iomega; vars.GetMiomega(m_iomega);
        Real_wp rho = vars.eval_rho0(iquad)(npoint);
        Real_wp c = vars.eval_c0(iquad)(npoint);
        coef = m_iomega*rho*c*nat_mat.GetCoefStiffness();
        for (int i = 0; i < Dimension::dim_N; i++)
          for (int j = 0; j < Dimension::dim_N; j++)
            Nabc(i+Dimension::dim_N, j)
              = coef*normale(i)*normale(j);

        coef = c/m_iomega*nat_mat.GetCoefStiffness();
        Nabc(2*Dimension::dim_N, 2*Dimension::dim_N) = coef;
      }
    }


  //! mutliplication by matrix N associated to the boundary condition
  /*!
    \param[in] normale outward normale
    \param[in] ref reference of the boundary
    \param[in] Vn vector to multiply
    \param[out] Un result Un = N Vn
    \param[in] num_elem1 element number
    \param[in] num_point local quadrature point number
    \param[in] nat_mat object containing mass and stiffness coefficients
    \param[in] ref2 reference of the element
    \param[in] vars given problem
    \param[in] Fb finite element associated with the element
    */
  template<class T, class Dimension>
    template<class Vector1, class TypeEquation, class T0>
    void GalbrunEquation_Base<T, Dimension>
    ::MltNabc(typename Dimension::R_N& normale, int ref, const Vector1& Vn,
        Vector1& Un, int num_elem1,
        int npoint, const GlobalGenericMatrix<T0>& nat_mat, int ref2, 
        const EllipticProblem<TypeEquation>& vars, const ElementReference<Dimension, 1>& Fb)
    {
      int cond = vars.mesh.GetBoundaryCondition(ref);
      Un.Fill(0);

      //TinyVector<Real_wp, Dimension::dim_N> v0 = vars.eval_flow(num_elem1)(npoint);

      if (cond == BoundaryConditionEnum::LINE_DIRICHLET)
      {
        for (int i = 0; i < Dimension::dim_N; i++)
        {
          Un(i+Dimension::dim_N) += -normale(i)*Vn(2*Dimension::dim_N)*vars.eval_rhoC2(num_elem1)(npoint);
          Un(2*Dimension::dim_N) += normale(i)*Vn(i);
        }

        Mlt(nat_mat.GetCoefStiffness(), Un);
      }
      else if (cond == BoundaryConditionEnum::LINE_NEUMANN)
      {
        for (int i = 0; i < Dimension::dim_N; i++)
        {
          Un(i+Dimension::dim_N) += normale(i)*Vn(2*Dimension::dim_N)*vars.eval_rhoC2(num_elem1)(npoint);
          Un(2*Dimension::dim_N) += -normale(i)*Vn(i);
        }

        Mlt(nat_mat.GetCoefStiffness(), Un);
      }
      else if (cond == BoundaryConditionEnum::LINE_ABSORBING)
      {
        T0 coef = nat_mat.GetCoefStiffness(), coef_p = coef;

        T m_iomega; vars.GetMiomega(m_iomega);
        Real_wp rho = vars.eval_rho0(num_elem1)(npoint);
        Real_wp c = vars.eval_c0(num_elem1)(npoint);

        coef *= m_iomega*rho*c;
        typename Vector1::value_type v_dot_n = 0;
        for (int j = 0; j < Dimension::dim_N; j++)
          v_dot_n += normale(j)*Vn(j);

        for (int i = 0; i < Dimension::dim_N; i++)
          Un(i+Dimension::dim_N) = coef*normale(i)*v_dot_n;

        coef_p *= c/m_iomega;
        Un(2*Dimension::dim_N) = coef_p*Vn(2*Dimension::dim_N);
      }
    }


  //! Computation of "penalization" matrices C
  /*!
    \param[out] Nabc penalization matrix C
    \param[in] normale outward normale
    \param[in] iquad element number
    \param[in] k local quadrature point number
    \param[in] nat_mat object containing mass and stiffness coefficients
    \param[in] ref reference of the boundary
    \param[in] vars given problem
    \param[in] Fb finite element associated with the element
    */
  template<class T, class Dimension>
    template<class Matrix1, class GenericPb, class T0>
    void GalbrunEquation_Base<T, Dimension>
    ::GetPenalDG(Matrix1& Nabc, typename Dimension::R_N& normale, int iquad, int k,
        int nf, const GlobalGenericMatrix<T0>& nat_mat, int ref, int ref2,
        const GenericPb& vars, const ElementReference<Dimension, 1>& Fb)
    {
      if (vars.upwind_fluxes)
      {
        T m_iomega; vars.GetMiomega(m_iomega);
        Matrix1 M, D, invM;
        GetTensorMass(vars, iquad, k, nat_mat, ref, M);
        M *= -m_iomega;
        GetInverse(M, invM);
        Mlt(invM, Nabc, D);

        GetAbsoluteValue(D, false);
        Mlt(M, D, Nabc);

        Nabc *= -nat_mat.GetCoefStiffness();
        return;
      }

      Nabc.Zero();
      T0 coef = vars.delta_penalization*nat_mat.GetCoefStiffness();
      T0 coef_p = vars.alpha_penalization*nat_mat.GetCoefStiffness();
      T m_iomega; vars.GetMiomega(m_iomega);
      Real_wp rho = vars.eval_rho0(iquad)(k);
      Real_wp c = vars.eval_c0(iquad)(k);

      coef *= m_iomega*rho*c;
      for (int i = 0; i < Dimension::dim_N; i++)
        for (int j = 0; j < Dimension::dim_N; j++)
          Nabc(i+Dimension::dim_N, j) = coef*normale(i)*normale(j);

      coef_p *= c/m_iomega;
      Nabc(2*Dimension::dim_N, 2*Dimension::dim_N) = coef_p;
    }


  //! Multiplication by penalization matrices
  /*!
    \param[in] normale outward normale
    \param[in] Vn vector to multiply
    \param[out] Un result vector Un = C*Vn
    \param[in] iquad element number
    \param[in] k local quadrature point number    
    \param[in] nat_mat object containing mass and stiffness coefficients
    \param[in] ref reference of the boundary
    \param[in] vars given problem
    \param[in] Fb finite element associated with the element
    */
  template<class T, class Dimension>
    template<class Vector1, class Vector2, class GenericPb, class T0>
    void GalbrunEquation_Base<T, Dimension>
    ::MltPenalDG(const typename Dimension::R_N& normale, const Vector1& Vn, Vector2& Un,
        int iquad, int k, int nf, const GlobalGenericMatrix<T0>& nat_mat,
        int ref, int ref2, const GenericPb& vars, const ElementReference<Dimension, 1>& Fb)
    {
      Un.Zero();
      T0 coef = vars.delta_penalization*nat_mat.GetCoefStiffness();
      T0 coef_p = vars.alpha_penalization*nat_mat.GetCoefStiffness();
      T m_iomega; vars.GetMiomega(m_iomega);
      Real_wp rho = vars.eval_rho0(iquad)(k);
      Real_wp c = vars.eval_c0(iquad)(k);

      coef *= m_iomega*rho*c;
      typename Vector1::value_type v_dot_n = 0;
      for (int j = 0; j < Dimension::dim_N; j++)
        v_dot_n += normale(j)*Vn(j);

      for (int i = 0; i < Dimension::dim_N; i++)
        Un(i+Dimension::dim_N) = coef*normale(i)*v_dot_n;

      coef_p *= c/m_iomega;
      Un(2*Dimension::dim_N) = coef_p*Vn(2*Dimension::dim_N);
    }


  /**************************
   * GalbrunEquationDG_Base *
   **************************/


  template<class T, class Dimension>
    void GalbrunEquationDG_Base<T, Dimension>
    ::SetIndexToCompute(VarGalbrunIndex_Base<Dimension>& var)
    {
      var.compute_div_flow = false;
      var.compute_grad_flow = true;
      var.compute_grad_rho = true;
      var.compute_grad_sigma = true;
      var.compute_grad_c0 = false;
      var.compute_hessian_flow = true;
      var.compute_hessian_p0 = true;
      var.compute_gamma = false;
      var.compute_source_momentum = true;
      var.compute_grad_momentum = false; //AJOUT NATHAN
    }


  //! for compatbility purpose
  template<class T, class Dimension>
    template<class TypeEquation>
    void GalbrunEquationDG_Base<T, Dimension>
    ::ComputeMassMatrix(EllipticProblem<TypeEquation>& var,
        int num_elem, const ElementReference<Dimension, 1>& Fb)
    {
    }


  //! which derivatives to evaluate during matrix-vector product ?
  template<class T, class Dimension>
    template<class GenericPb, class T0, class Vector1>
    void GalbrunEquationDG_Base<T, Dimension>
    ::GetNeededDerivative(const GenericPb& vars, const GlobalGenericMatrix<T0>& nat_mat,
        Vector1& unknown_to_derive, Vector1& fct_test_to_derive)
    {
      fct_test_to_derive.Fill(true);
      unknown_to_derive.Fill(true);
    }


  //! fills tensors D and E appearing in the variational formulation
  /*!
    \param[in] vars considered problem
    \param[in] num_elem element where D and E must be computed
    \param[in] jloc quadrature point where D and E must be computed
    \param[in] nat_mat object containing mass and stiffness coefficients
    \param[in] ref reference of the element
    \param[out] Ephi_grad tensor E
    \param[out] Dphi_grad tensor D    
    The tensors D and E are appearing in the terms
    \int_K E \nabla u v + D u \nabla v dx
    of the variational formulation
    */    
  template<class T, class Dimension>
    template<class TypeEquation, class T0, class MatStiff>
    void GalbrunEquationDG_Base<T, Dimension>
    ::GetGradPhiTensor(const EllipticProblem<TypeEquation>& vars,
        int num_elem, int jloc, const GlobalGenericMatrix<T0>& nat_mat, int ref,
        MatStiff& Dgrad_phi, MatStiff& Ephi_grad)
    {
      FillZero(Dgrad_phi);
      FillZero(Ephi_grad);

      TinyVector<Real_wp, Dimension::dim_N> v0 = vars.eval_flow(num_elem)(jloc);

      Real_wp rho = vars.eval_rho0(num_elem)(jloc);
      Real_wp coef = DotProd(v0, vars.grad_rho0(num_elem)(jloc))/rho;
      for (int i = 0; i < Dimension::dim_N; i++)
      {
        // acoustic part
        // -rho^2 c^2 div u in equation of p
        Ephi_grad(2*Dimension::dim_N, i)(i) = -rho*vars.eval_rhoC2(num_elem)(jloc);
        // term -grad(p) in equation of u
        //Ephi_grad(i, 2*Dimension::dim_N)(i) = -1.0; 
        // this term is put in D to have a skew-symmetric matrix for null flow and approximate integration
        Dgrad_phi(i, 2*Dimension::dim_N)(i) = 1.0; 

        // flow part
        // terms rho M \cdot \nabla in equations of u, q and p
        if (vars.drop_unstable_terms != vars.DROP_CONVECTIVE)
          for (int p = 0; p <= 2*Dimension::dim_N; p++)
            Ephi_grad(p, p)(i) += rho*v0(i);
        else
        {
          for (int p = 0; p < Dimension::dim_N; p++)
            Ephi_grad(p, p)(i) += rho*v0(i);

          for (int p = Dimension::dim_N; p < 2*Dimension::dim_N; p++)
            Ephi_grad(p, p)(i) += vars.coef_convective_term*rho*v0(i);

          Ephi_grad(2*Dimension::dim_N, 2*Dimension::dim_N)(i) += rho*v0(i);
        }

        if (vars.drop_unstable_terms != vars.DROP_NON_UNIFORM)
        {
          for (int p = 0; p < Dimension::dim_N; p++)
          {
            // term (div u) grad(p_0) in equation of q
            Ephi_grad(Dimension::dim_N+i, p)(p)
              += vars.grad_p0(num_elem)(jloc)(i);

            // AJOUT NATHAN - part -div(u)f_0
            if(vars.compute_source_momentum)
              Ephi_grad(Dimension::dim_N+i, p)(p)
                -= vars.eval_f0(num_elem)(jloc)(i);

            // part - (grad u)^T grad(p_0) in equation of q
            Ephi_grad(Dimension::dim_N+i, p)(i)
              -= vars.grad_p0(num_elem)(jloc)(p);
          }

          // gradient of flow
          // term - (grad M)^T \grad p in equation of q
          for (int j = 0; j < Dimension::dim_N; j++)
            Ephi_grad(j+Dimension::dim_N, 2*Dimension::dim_N)(i) = -vars.grad_flow(num_elem)(jloc)(i, j);

          // term - M \cdot grad(rho)/rho grad(p) in equation of q
          Ephi_grad(Dimension::dim_N+i, 2*Dimension::dim_N)(i) -= coef;          
        }
      }

      Ephi_grad *= nat_mat.GetCoefStiffness();    
    }


  //! Applying the tensor E to grad(v)
  /*!
    \param[in] var problem to be solved
    \param[in] i element number
    \param[in] j quadrature point number inside the element
    \param[in] nat_mat mass and stiffness coefficients
    \param[in] ref reference of the physical domain
    \param[in] Vn gradient of the unknown vector V
    \param[out] Un result to E grad(v)
    */  
  template<class T, class Dimension>
    template<class TypeEquation, class T0, class Vector1, class Vector2>
    void GalbrunEquationDG_Base<T, Dimension>
    ::ApplyGradientUnknown(const EllipticProblem<TypeEquation>& vars,
        int iquad, int k, const GlobalGenericMatrix<T0>& nat_mat,
        int ref, Vector1& Vn, Vector2& Un)
    {
      Un.Fill(0);
      TinyVector<Real_wp, Dimension::dim_N> v0 = vars.eval_flow(iquad)(k);

      Real_wp rho = vars.eval_rho0(iquad)(k);
      Real_wp coef = DotProd(v0, vars.grad_rho0(iquad)(k))/rho;
      for (int i = 0; i < Dimension::dim_N; i++)
      {
        // flow part
        // terms rho M \cdot \nabla in equations of u, q and p
        if (vars.drop_unstable_terms != vars.DROP_CONVECTIVE)
          for (int p = 0; p <= 2*Dimension::dim_N; p++)
            Un(p) += rho*v0(i)*Vn(p)(i);
        else
        {
          for (int p = 0; p < Dimension::dim_N; p++)
            Un(p) += rho*v0(i)*Vn(p)(i);

          for (int p = Dimension::dim_N; p < 2*Dimension::dim_N; p++)
            Un(p) += vars.coef_convective_term*rho*v0(i)*Vn(p)(i);

          Un(2*Dimension::dim_N) += rho*v0(i)*Vn(2*Dimension::dim_N)(i);
        }

        // acoustic part
        // term -rho^2 c^2 div u in equation of p
        Un(2*Dimension::dim_N) -= rho*vars.eval_rhoC2(iquad)(k)*Vn(i)(i);
        // term -grad(p) in equation of u
        Un(i) -= Vn(2*Dimension::dim_N)(i);	

        if (vars.drop_unstable_terms != vars.DROP_NON_UNIFORM)
        {
          for (int p = 0; p < Dimension::dim_N; p++)
          {
            // term (div u) grad(p_0) in equation of q
            Un(Dimension::dim_N+i) += vars.grad_p0(iquad)(k)(i)*Vn(p)(p);
            // AJOUT NATHAN
            if(vars.compute_source_momentum)
              Un(Dimension::dim_N+i) -= vars.eval_f0(iquad)(k)(i)*Vn(p)(p);

            // part - (grad u)^T grad(p_0) in equation of q
            Un(Dimension::dim_N+i) -= vars.grad_p0(iquad)(k)(p)*Vn(p)(i);
          }

          // gradient of flow
          // term - (grad M)^T \grad p in equation of q
          for (int j = 0; j < Dimension::dim_N; j++)
            Un(Dimension::dim_N+j) -= vars.grad_flow(iquad)(k)(i, j)*Vn(2*Dimension::dim_N)(i);

          // term - M \cdot grad(rho)/rho grad(p) in equation of q
          Un(Dimension::dim_N+i) -= coef*Vn(2*Dimension::dim_N)(i);
        }
      }

      Mlt(nat_mat.GetCoefStiffness(), Un);
    }


  //! Applies matrix A to a vector
  /*!
    \param[in] var given problem
    \param[in] iquad number of the element where M needs to be evaluated
    \param[in] k number of the local quadrature point in the element
    \param[in] nat_mat mass and stiffness coefficients
    \param[in] ref reference of the physical domain
    \param[in] Un vector to be multiplied by A
    \param[out] Vn result vector Vn = A Un
    */  
  template<class T, class Dimension>
    template<class TypeEquation, class T0, class Vector1>
    void GalbrunEquationDG_Base<T, Dimension>
    ::ApplyTensorMass(const EllipticProblem<TypeEquation>& var,
        int iquad, int k, const GlobalGenericMatrix<T0>& nat_mat,
        int ref, Vector1& Un, Vector1& Vn)
    {
      Vn.Fill(0);
      Real_wp rho = var.eval_rho0(iquad)(k);
      for (int i = 0; i < Dimension::dim_N; i++)
      {
        // term -rho q in equation of u
        Vn(i) = -rho*Un(i+Dimension::dim_N);

        // term -grad(sigma) p in equation of q
        Vn(Dimension::dim_N+i) -= var.grad_sigma(iquad)(k)(i)*Un(2*Dimension::dim_N);

        //AJOUT NATHAN -- Term -(\nabla f)u in equation of q
        if(var.compute_grad_momentum){
          for(int l=0; l<Dimension::dim_N; l++){
            Vn(Dimension::dim_N+i) -= var.grad_f0(iquad)(k)(i,l)*Un(i);
          }
        }
      }

      Mlt(nat_mat.GetCoefStiffness(), Vn);

      Real_wp sigma = var.eval_sigma(iquad)(k);
      T m_iomega; var.GetMiomega(m_iomega);
      T0 coef = m_iomega*nat_mat.GetCoefMass() + sigma*nat_mat.GetCoefDamping(); 

      coef *= rho;
      for (int p = 0; p <= 2*Dimension::dim_N; p++)
        Vn(p) += coef*Un(p);
    }


  //! returns the matrix A, in the integral \f$ \int_K A u v \f$ 
  /*!
    \param[in] vars given problem
    \param[in] iquad number of the element where A needs to be evaluated
    \param[in] k number of the local quadrature point in the element
    \param[in] nat_mat mass and stiffness coefficients
    \param[in] ref reference of the physical domain
    \param[out] Cj matrix A
    */  
  template<class T, class Dimension>
    template<class TypeEquation, class T0, class MatMass>
    void GalbrunEquationDG_Base<T, Dimension>
    ::GetTensorMass(const EllipticProblem<TypeEquation>& vars,
        int iquad, int k, const GlobalGenericMatrix<T0>& nat_mat, int ref, MatMass& Cj)
    {
      Cj.Fill(0);    
      Real_wp rho = vars.eval_rho0(iquad)(k);
      Real_wp sigma = vars.eval_sigma(iquad)(k);
      for (int i = 0; i < Dimension::dim_N; i++)
      {
        // term -rho q in equation of u
        Cj(i, i+Dimension::dim_N) = -rho*nat_mat.GetCoefStiffness();    

        // term -grad(sigma) p in equation of q
        Cj(i+Dimension::dim_N, 2*Dimension::dim_N) = -vars.grad_sigma(iquad)(k)(i)*nat_mat.GetCoefStiffness();

        //AJOUT NATHAN -- Term -(\nabla f)u in equation of q
        if(vars.compute_grad_momentum){
          for(int l=0; l<Dimension::dim_N; l++){
            Cj(i+Dimension::dim_N, l) = -vars.grad_f0(iquad)(k)(i,l);
          }
        }
      }

      T m_iomega; vars.GetMiomega(m_iomega);
      T0 coef = m_iomega*nat_mat.GetCoefMass() + sigma*nat_mat.GetCoefDamping(); 

      coef *= rho;
      for (int p = 0; p <= 2*Dimension::dim_N; p++)
        Cj(p, p) += coef;
    }


  //! fills matrix |D| from matrix D
  template<class T, class Dimension> template<class T0>
    void GalbrunEquationDG_Base<T, Dimension>::
    GetAbsoluteD(TinyMatrix<T0, General, 5, 5>& Dtest, const R2& normale,
        const Real_wp& c0, bool non_unif, const Real_wp& alpha1_)
    {
      // 2-D case
      Real_wp alpha = realpart(Dtest(0, 0));
      Real_wp nx = normale(0), ny = normale(1);
      Real_wp a = abs(alpha), d = 0.5*(abs(alpha+c0) - abs(alpha-c0)), s = 0.5*(abs(alpha+c0) + abs(alpha-c0));

      if (alpha > c0)
      {
        // nothing to do, Dtest is already positive
        // Dtest = Dtest;
      }
      else if (alpha < -c0)
        Dtest *= -1.0;
      else
      {
        a = abs(alpha); d = 0.5*(abs(alpha+c0) - abs(alpha-c0)); s = 0.5*(abs(alpha+c0) + abs(alpha-c0));
        Dtest(0, 0) = a + nx*nx*(s-a); Dtest(0, 1) = nx*ny*(s-a); Dtest(0, 2) = 0; Dtest(0, 3) = 0; Dtest(0, 4) = -nx*d/c0;
        Dtest(1, 0) = nx*ny*(s-a); Dtest(1, 1) = a + ny*ny*(s-a); Dtest(1, 2) = 0; Dtest(1, 3) = 0; Dtest(1, 4) = -ny*d/c0;
        if (non_unif)
        {            
          Real_wp gamma = realpart(Dtest(2, 1)), bx = realpart(Dtest(2, 4)), by = realpart(Dtest(3, 4));

          if (alpha1_ != Real_wp(0))
          {
            Real_wp alpha1 = alpha1_*alpha;
            Real_wp a11 = (-gamma*ny + bx*c0) / (alpha + c0 - alpha1);
            Real_wp a12 = (gamma*ny + bx*c0) / (alpha - c0 - alpha1);
            Real_wp a21 = (gamma*nx + by*c0) / (alpha + c0 - alpha1);
            Real_wp a22 = (-gamma*nx + by*c0) / (alpha - c0 - alpha1);

            Real_wp sa = sign(alpha), sAc = (abs(alpha1) - abs(alpha+c0))/2, dAc = (abs(alpha-c0)-abs(alpha1))/2;
            Dtest(2, 0) = -gamma*nx*ny*sa + nx*(a11*sAc + a12*dAc);
            Dtest(2, 1) = gamma*nx*nx*sa + ny*(a11*sAc + a12*dAc);
            Dtest(2, 2) = abs(alpha1); Dtest(2, 3) = 0; Dtest(2, 4) = (-a11*sAc + a12*dAc)/c0;
            Dtest(3, 0) = -gamma*ny*ny*sa + nx*(a21*sAc + a22*dAc);
            Dtest(3, 1) = gamma*nx*ny*sa + ny*(a21*sAc + a22*dAc);
            Dtest(3, 2) = 0; Dtest(3, 3) = abs(alpha1); Dtest(3, 4) = (-a21*sAc + a22*dAc)/c0;                
          }
          else
          {
            Real_wp sa = sign(alpha), sAc = 0.5*(sign(alpha+c0)+sign(alpha-c0)), dAc = 0.5*(sign(alpha+c0)-sign(alpha-c0));
            Dtest(2, 0) = nx*ny*gamma*(-sa+sAc)-nx*bx*c0*dAc;
            Dtest(2, 1) = gamma*nx*nx*sa+ny*ny*gamma*sAc-bx*ny*c0*dAc;
            Dtest(2, 2) = 0; Dtest(2, 3) = 0; Dtest(2, 4) = -gamma*ny/c0*dAc+bx*sAc;
            Dtest(3, 0) = -ny*ny*gamma*sa-nx*nx*gamma*sAc-by*nx*c0*dAc;
            Dtest(3, 1) = nx*ny*gamma*(sa-sAc)-by*ny*c0*dAc;
            Dtest(3, 2) = 0; Dtest(3, 3) = 0; Dtest(3, 4) = gamma*nx/c0*dAc+by*sAc;
          }
        }
        else
        {
          Dtest(2, 0) = 0; Dtest(2, 1) = 0; Dtest(2, 2) = a; Dtest(2, 3) = 0; Dtest(2, 4) = 0;
          Dtest(3, 0) = 0; Dtest(3, 1) = 0; Dtest(3, 2) = 0; Dtest(3, 3) = a; Dtest(3, 4) = 0;
        }

        Dtest(4, 0) = -nx*d*c0; Dtest(4, 1) = -ny*d*c0; Dtest(4, 2) = 0; Dtest(4, 3) = 0; Dtest(4, 4) = s;
      }
    }


  //! fills matrix |D| from matrix D
  template<class T, class Dimension> template<class T0>
    void GalbrunEquationDG_Base<T, Dimension>::
    GetAbsoluteD(TinyMatrix<T0, General, 7, 7>& Dtest, const R3& normale, const Real_wp& c0, bool non_unif, const Real_wp& alpha1_)
    {
      // 3-D case
      Real_wp alpha = realpart(Dtest(0, 0));
      Real_wp nx = normale(0), ny = normale(1), nz = normale(2);
      Real_wp a = abs(alpha), d = 0.5*(abs(alpha+c0) - abs(alpha-c0)), s = 0.5*(abs(alpha+c0) + abs(alpha-c0));

      if (alpha > c0)
      {
        // nothing to do, Dtest is already positive
        // Dtest = Dtest
      }
      else if (alpha < -c0)
        Dtest *= -1.0;
      else
      {
        a = abs(alpha); d = 0.5*(abs(alpha+c0) - abs(alpha-c0)); s = 0.5*(abs(alpha+c0) + abs(alpha-c0));
        Dtest(0, 0) = a + nx*nx*(s-a); Dtest(0, 1) = nx*ny*(s-a); Dtest(0, 2) = nx*nz*(s-a); Dtest(0, 3) = 0; Dtest(0, 4) = 0; Dtest(0, 5) = 0; Dtest(0, 6) = -nx*d/c0;
        Dtest(1, 0) = nx*ny*(s-a); Dtest(1, 1) = a + ny*ny*(s-a); Dtest(1, 2) = ny*nz*(s-a); Dtest(1, 3) = 0; Dtest(1, 4) = 0; Dtest(1, 5) = 0; Dtest(1, 6) = -ny*d/c0;
        Dtest(2, 0) = nx*nz*(s-a); Dtest(2, 1) = ny*nz*(s-a); Dtest(2, 2) = a + nz*nz*(s-a); Dtest(2, 3) = 0; Dtest(2, 4) = 0; Dtest(2, 5) = 0; Dtest(2, 6) = -nz*d/c0;
        if (non_unif)
        {
          cout << "not implemented" << endl;
          abort();
        }
        else
        {
          Dtest(3, 0) = 0; Dtest(3, 1) = 0; Dtest(3, 2) = 0; Dtest(3, 3) = a; Dtest(3, 4) = 0; Dtest(3, 5) = 0; Dtest(3, 6) = 0;
          Dtest(4, 0) = 0; Dtest(4, 1) = 0; Dtest(4, 2) = 0; Dtest(4, 3) = 0; Dtest(4, 4) = a; Dtest(4, 5) = 0; Dtest(4, 6) = 0;
          Dtest(5, 0) = 0; Dtest(5, 1) = 0; Dtest(5, 2) = 0; Dtest(5, 3) = 0; Dtest(5, 4) = 0; Dtest(5, 5) = a; Dtest(5, 6) = 0;
        }

        Dtest(6, 0) = -nx*d*c0; Dtest(6, 1) = -ny*d*c0; Dtest(6, 2) = -nz*d*c0; Dtest(6, 3) = 0; Dtest(6, 4) = 0; Dtest(6, 5) = 0; Dtest(6, 6) = s;
      }
    }


  template<class T, class Dimension> template<class T0, class GenericPb>
    void GalbrunEquationDG_Base<T, Dimension>::
    GetAbsoluteMatrixD(TinyMatrix<T0, General, 5, 5>& D, const R2& normale, int iquad, int k,
        const GlobalGenericMatrix<T0>& nat_mat, const GenericPb& vars)
    {
      Real_wp alpha = DotProd(vars.eval_flow(iquad)(k), normale);
      Real_wp rho0 = vars.eval_rho0(iquad)(k);
      Real_wp c0 = vars.eval_c0(iquad)(k);
      Real_wp alpha1 = 0;

      if (vars.drop_unstable_terms != vars.DROP_NON_UNIFORM)
      {     
        Real_wp gamma = vars.grad_p0(iquad)(k)(0)*normale(1) - vars.grad_p0(iquad)(k)(1)*normale(0);
        Real_wp coef = DotProd(vars.eval_flow(iquad)(k), vars.grad_rho0(iquad)(k))/rho0;
        typename Dimension::R_N vec_u;
        MltTrans(vars.grad_flow(iquad)(k), normale, vec_u);
        Real_wp bx = -coef*normale(0)-vec_u(0);
        Real_wp by = -coef*normale(1)-vec_u(1);

        D(2, 0) = 0; D(2, 1) = gamma; D(2, 4) = bx;
        D(3, 0) = -gamma; D(3, 1) = 0; D(3, 4) = by;
      }

      if (vars.drop_unstable_terms == vars.DROP_CONVECTIVE)
      {
        alpha1 = vars.coef_convective_term*alpha;
        D(2, 2) = rho0*alpha1; D(2, 3) = 0; 
        D(3, 2) = 0; D(3, 3) = rho0*alpha1;
      }
      else
      {
        D(2, 2) = rho0*alpha; D(2, 3) = 0;
        D(3, 2) = 0; D(3, 3) = rho0*alpha;
      }

      Real_wp nx = normale(0), ny = normale(1);
      D(0, 0) = rho0*alpha; D(0, 1) = 0; D(0, 2) = 0; D(0, 3) = 0; D(0, 4) = -nx;
      D(1, 0) = 0; D(1, 1) = rho0*alpha; D(1, 2) = 0; D(1, 3) = 0; D(1, 4) = -ny;
      D(4, 0) = -square(rho0*c0)*nx; D(4, 1) = -square(rho0*c0)*ny;
      D(4, 2) = 0; D(4, 3) = 0; D(4, 4) = rho0*alpha;

      bool non_unif = false;
      if ((vars.drop_unstable_terms == vars.DROP_CONVECTIVE)
          || (vars.drop_unstable_terms == vars.DROP_NONE))
        non_unif = true;

      GetAbsoluteD(D, normale, c0*rho0, non_unif, alpha1);
    }


  template<class T, class Dimension> template<class T0, class GenericPb>
    void GalbrunEquationDG_Base<T, Dimension>::
    GetAbsoluteMatrixD(TinyMatrix<T0, General, 7, 7>& D, const R3& normale, int iquad, int k,
        const GlobalGenericMatrix<T0>& nat_mat, const GenericPb& vars)
    {
      cout << "Not implemented in 3-D" << endl;
      abort();
    }


  //! computation of matrix N associated to the boundary condition
  /*!
    \param[out] Nabc matrix N
    \param[in] normale outward normale
    \param[in] ref reference of the boundary
    \param[in] iquad element number
    \param[in] npoint local quadrature point number
    \param[in] nat_mat object containing mass and stiffness coefficients
    \param[in] ref2 reference of the element
    \param[in] vars considered problem
    \param[in] Fb finite element associated with the element
    */  
  template<class T, class Dimension>
    template<class Matrix1, class GenericPb, class T0>
    void GalbrunEquationDG_Base<T, Dimension>
    ::GetNabc(Matrix1& Nabc, const typename Dimension::R_N& normale,
        int ref, int iquad, int npoint,
        const GlobalGenericMatrix<T0>& nat_mat, int ref2, 
        const GenericPb& vars, const ElementReference<Dimension, 1>& Fb)
    {
      int cond = vars.mesh.GetBoundaryCondition(ref);

      typename Dimension::R_N v0 = vars.eval_flow(iquad)(npoint);

      if (cond == BoundaryConditionEnum::LINE_DIRICHLET)
      {
        Nabc.Fill(0);
        Real_wp rho = vars.eval_rho0(iquad)(npoint);
        Real_wp rhoC = rho*vars.eval_rhoC2(iquad)(npoint);
        Real_wp coef = DotProd(v0, vars.grad_rho0(iquad)(npoint))/rho;
        typename Dimension::R_N vec_u;
        MltTrans(vars.grad_flow(iquad)(npoint), normale, vec_u);
        for (int i = 0; i < Dimension::dim_N; i++)
        {
          Nabc(2*Dimension::dim_N, i) = rhoC*normale(i);
          Nabc(i, 2*Dimension::dim_N) = -normale(i);

          Nabc(i+Dimension::dim_N, 2*Dimension::dim_N) 
            = -vec_u(i)-coef*normale(i);
        }

        Mlt(nat_mat.GetCoefStiffness(), Nabc);
      }
      else if (cond == BoundaryConditionEnum::LINE_NEUMANN)
      {
        Nabc.Fill(0);
        Real_wp rho = vars.eval_rho0(iquad)(npoint);
        Real_wp rhoC = rho*vars.eval_rhoC2(iquad)(npoint);
        Real_wp coef = DotProd(v0, vars.grad_rho0(iquad)(npoint))/rho;
        typename Dimension::R_N vec_u;
        MltTrans(vars.grad_flow(iquad)(npoint), normale, vec_u);
        for (int i = 0; i < Dimension::dim_N; i++)
        {
          Nabc(2*Dimension::dim_N, i) = -rhoC*normale(i);
          Nabc(i, 2*Dimension::dim_N) = normale(i);

          Nabc(i+Dimension::dim_N, 2*Dimension::dim_N) 
            = vec_u(i)+coef*normale(i);
        }

        Mlt(nat_mat.GetCoefStiffness(), Nabc);
      }
      else if (cond == BoundaryConditionEnum::LINE_ABSORBING)
      {
        Real_wp c0 = vars.eval_c0(iquad)(npoint);
        c0 *= vars.eval_rho0(iquad)(npoint);
        bool non_unif = false;
        if ((vars.drop_unstable_terms == vars.DROP_CONVECTIVE)
            || (vars.drop_unstable_terms == vars.DROP_NONE))
          non_unif = true;

        GetAbsoluteD(Nabc, normale, c0, non_unif, vars.coef_convective_term);
      }
    }


  //! mutliplication by matrix N associated to the boundary condition
  /*!
    \param[in] normale outward normale
    \param[in] ref reference of the boundary
    \param[in] Vn vector to multiply
    \param[out] Un result Un = N Vn
    \param[in] num_elem1 element number
    \param[in] num_point local quadrature point number
    \param[in] nat_mat object containing mass and stiffness coefficients
    \param[in] ref2 reference of the element
    \param[in] vars given problem
    \param[in] Fb finite element associated with the element
    */
  template<class T, class Dimension>
    template<class Vector1, class TypeEquation, class T0>
    void GalbrunEquationDG_Base<T, Dimension>
    ::MltNabc(typename Dimension::R_N& normale, int ref,
        const Vector1& Vn, Vector1& Un, int num_elem1,
        int npoint, const GlobalGenericMatrix<T0>& nat_mat, int ref2, 
        const EllipticProblem<TypeEquation>& vars, const ElementReference<Dimension, 1>& Fb)
    {
      int cond = vars.mesh.GetBoundaryCondition(ref);
      Un.Fill(0);
      typename Dimension::R_N v0 = vars.eval_flow(num_elem1)(npoint);

      if (cond == BoundaryConditionEnum::LINE_DIRICHLET)
      {
        Real_wp rho = vars.eval_rho0(num_elem1)(npoint);
        Real_wp rhoC = rho*vars.eval_rhoC2(num_elem1)(npoint);
        Real_wp coef = DotProd(v0, vars.grad_rho0(num_elem1)(npoint))/rho;
        typename Dimension::R_N vec_u;
        MltTrans(vars.grad_flow(num_elem1)(npoint), normale, vec_u);
        for (int i = 0; i < Dimension::dim_N; i++)
        {
          Un(2*Dimension::dim_N) += rhoC*normale(i)*Vn(i);
          Un(i) += -normale(i)*Vn(2*Dimension::dim_N);

          Un(i+Dimension::dim_N) -= (vec_u(i)+coef*normale(i))*Vn(2*Dimension::dim_N);
        }

        Mlt(nat_mat.GetCoefStiffness(), Un);
      }
      else if (cond == BoundaryConditionEnum::LINE_NEUMANN)
      {
        Real_wp rho = vars.eval_rho0(num_elem1)(npoint);
        Real_wp rhoC = rho*vars.eval_rhoC2(num_elem1)(npoint);
        Real_wp coef = DotProd(v0, vars.grad_rho0(num_elem1)(npoint))/rho;
        typename Dimension::R_N vec_u;
        MltTrans(vars.grad_flow(num_elem1)(npoint), normale, vec_u);
        for (int i = 0; i < Dimension::dim_N; i++)
        {
          Un(2*Dimension::dim_N) += rhoC*normale(i)*Vn(i);
          Un(i) += -normale(i)*Vn(2*Dimension::dim_N);

          Un(i+Dimension::dim_N) -= (vec_u(i)+coef*normale(i))*Vn(2*Dimension::dim_N);
        }

        Mlt(-nat_mat.GetCoefStiffness(), Un);
      }
      else if (cond == BoundaryConditionEnum::LINE_ABSORBING)
      {
        TinyMatrix<T0, General, nb_unknowns, nb_unknowns> D;
        GetAbsoluteMatrixD(D, normale, num_elem1, npoint, nat_mat, vars);

        Mlt(D, Vn, Un);

        Un *= nat_mat.GetCoefStiffness();
      }
    }


  //! Computation of "penalization" matrices C
  /*!
    \param[out] Nabc penalization matrix C
    \param[in] normale outward normale
    \param[in] iquad element number
    \param[in] npoint local quadrature point number
    \param[in] nat_mat object containing mass and stiffness coefficients
    \param[in] ref reference of the boundary
    \param[in] vars given problem
    \param[in] Fb finite element associated with the element
    */
  template<class T, class Dimension>
    template<class Matrix1, class GenericPb, class T0>
    void GalbrunEquationDG_Base<T, Dimension>
    ::GetPenalDG(Matrix1& Nabc, typename Dimension::R_N& normale, int iquad, int npoint,
        int nf, const GlobalGenericMatrix<T0>& nat_mat, int ref, int ref2,
        const GenericPb& vars, const ElementReference<Dimension, 1>& Fb)
    {
      Real_wp rho = vars.eval_rho0(iquad)(npoint);
      Real_wp c = vars.eval_c0(iquad)(npoint);

      if (vars.upwind_fluxes)
      {
        //Matrix1 Nabc_ref(Nabc);
        //GetAbsoluteValue(Nabc_ref); DISP(Nabc_ref);

        /* T m_iomega; vars.GetMiomega(m_iomega);
           Matrix1 M, D, invM;
           GetTensorMass(vars, iquad, npoint, nat_mat, ref, M);
           M *= -m_iomega;
           GetInverse(M, invM);
           Mlt(invM, Nabc, D);

           GetAbsoluteValue(D, true);
           Mlt(M, D, Nabc);
           */
        bool non_unif = false;
        if ((vars.drop_unstable_terms == vars.DROP_CONVECTIVE)
            || (vars.drop_unstable_terms == vars.DROP_NONE))
          non_unif = true;

        GetAbsoluteD(Nabc, normale, rho*c, non_unif, vars.coef_convective_term);

        /*DISP(Nabc); DISP(Nabc_ref);
          for (int i = 0; i < Nabc.GetM(); i++)
          for (int j = 0; j < Nabc.GetM(); j++)
          if (abs(Nabc(i, j) - Nabc_ref(i, j)) > 1e-10 || isnan(Nabc(i, j)) || isnan(Nabc_ref(i, j)))
          {
          DISP(i); DISP(j); DISP(Nabc); DISP(Nabc_ref);
          abort();
          }
          */
        Nabc *= -nat_mat.GetCoefStiffness();
        return;
      }

      Nabc.Zero();
      T0 coef;
      coef = rho*c*nat_mat.GetCoefStiffness();

      Nabc(2*Dimension::dim_N, 2*Dimension::dim_N) = coef*vars.alpha_penalization;
      coef *= vars.delta_penalization;
      for (int i = 0; i < Dimension::dim_N; i++)
        for (int j = 0; j < Dimension::dim_N; j++)
          Nabc(i, j) = coef*normale(i)*normale(j);
    }


  //! Multiplication by penalization matrices
  /*!
    \param[in] normale outward normale
    \param[in] Vn vector to multiply
    \param[out] Un result vector Un = C*Vn
    \param[in] iquad element number
    \param[in] npoint local quadrature point number    
    \param[in] nat_mat object containing mass and stiffness coefficients
    \param[in] ref reference of the boundary
    \param[in] vars given problem
    \param[in] Fb finite element associated with the element
    */
  template<class T, class Dimension>
    template<class Vector1, class Vector2, class GenericPb, class T0>
    void GalbrunEquationDG_Base<T, Dimension>
    ::MltPenalDG(const typename Dimension::R_N& normale, const Vector1& Vn, Vector2& Un,
        int iquad, int npoint, int nf, const GlobalGenericMatrix<T0>& nat_mat,
        int ref, int ref2, const GenericPb& vars,
        const ElementReference<Dimension, 1>& Fb)
    {
      if (vars.upwind_fluxes)
      {
        TinyMatrix<T0, General, nb_unknowns, nb_unknowns> D;
        GetAbsoluteMatrixD(D, normale, iquad, npoint, nat_mat, vars);

        Mlt(D, Vn, Un);

        Un *= -nat_mat.GetCoefStiffness();
        return;
      }

      Un.Zero();

      T0 coef, v_dot_n;
      Real_wp rho = vars.eval_rho0(iquad)(npoint);
      Real_wp c = vars.eval_c0(iquad)(npoint);
      coef = rho*c*nat_mat.GetCoefStiffness();

      Un(2*Dimension::dim_N) = coef*Vn(2*Dimension::dim_N)*vars.alpha_penalization;

      coef *= vars.delta_penalization;
      v_dot_n = 0;
      for (int i = 0; i < Dimension::dim_N; i++)
        v_dot_n += Vn(i)*normale(i);

      for (int i = 0; i < Dimension::dim_N; i++)
        Un(i) = coef*v_dot_n*normale(i);
    }


  /************************
   * VarGalbrunIndex_Base *
   ************************/


  //! parameters of the data file, specific to Galbrun equation
  /*!
    \param[in] description_field keyword of the considered line of the data file
    \param[in] parameters list of values associated
    \param[in] nb_param number of values
    */
  template<class Dimension>
    void VarGalbrunIndex_Base<Dimension>
    ::SetInputData(const string& description_field, const VectString& parameters)
    {
      if (!description_field.compare("ForceFlowNeumann"))
      {
        if (parameters.GetM() <= 0)
        {
          cout << "In SetInputData of ForceFlowNeumann" << endl;
          cout << "ForceFlowNeumann needs at least one parameter, for instance :" << endl;
          cout << "ForceFlowNeumann = YES" << endl;
          cout << "Current parameters are : " << endl << parameters << endl;
          abort();
        }

        adjustment_neumann = false;
        if (!parameters(0).compare("YES"))
          adjustment_neumann = true;
      }
    }


  //! initialization of physical indices (rho0, p0 and v0)
  /*!
    \param[in] n number of different physical domains
    The steady flow is caracterized by
    rho0 the mass density, p0 the pressure and v0 the velocity
    */
  template<class Dimension>
    void VarGalbrunIndex_Base<Dimension>::InitIndices(int n)
    {
      ref_v0.Reallocate(n);
      ref_rho0.Reallocate(n);
      ref_p0.Reallocate(n);
      //    ref_f0.Reallocate(n); //AJOUT NATHAN
      ref_c0.Reallocate(n);
      ref_sigma.Reallocate(n);
      for (int i = 0; i < n; i++)
      {
        R_N v0; v0(0) = 0.0;
        //	R_N f0; f0(0) = 0.0;
        ref_v0(i).SetConstant(v0);
        //	ref_f0(i).SetConstant(f0); //AJOUT NATHAN
        ref_sigma(i).SetConstant(0.0);
        ref_rho0(i).SetConstant(Real_wp(1));
        ref_c0(i).SetConstant(Real_wp(1));
        ref_p0(i).SetConstant(Real_wp(1));
      }
    }


  //! returns the number of different physical media
  template<class Dimension>
    int VarGalbrunIndex_Base<Dimension>::GetNbPhysicalIndices() const
    {
      //    std::cout << "NUM PHYS INDICES :" << ref_v0.GetM() << std::endl;
      return ref_v0.GetM();
    }



  //! reading physical indices
  /*!
    \param[in] i physical domain domain
    \param[in] parameters parameters of the matching line of the data file
    the data file contains a line like MateriauDielec = ...
    */
  template<class Dimension>
    void VarGalbrunIndex_Base<Dimension>::SetIndices(int i, const VectString& parameters)
    {
      int nb = 1;
      if (i >= ref_rho0.GetM())
      {
        cout << "Not enough indices stored, call InitIndices with a higher N" << endl;
        cout << "Current reference i : " << i << " < " << ref_rho0.GetM() << endl;
        abort();
      }

      ref_v0(i).SetInputData(nb, parameters, parameters(0));
      ref_sigma(i).SetInputData(nb, parameters, parameters(0));
      ref_rho0(i).SetInputData(nb, parameters, parameters(0));
      ref_c0(i).SetInputData(nb, parameters, parameters(0));
      ref_p0(i).SetInputData(nb, parameters, parameters(0));
      //    ref_f0(i).SetInputData(nb, parameters, parameters(0)); //AJOUT NATHAN

    }


  //! reading of a physical index
  /*!
    \param[in] name_media name of the physical index
    \param[in] i physical domain domain
    \param[in] parameters parameters of the matching line of the data file
    the data file contains a line like PhysicalMedia = ...
    */
  template<class Dimension>
    void VarGalbrunIndex_Base<Dimension>
    ::SetPhysicalIndex(const string& name_media, int i, const VectString& parameters)
    {
      int nb = 1;
      if (i >= ref_rho0.GetM())
      {
        cout << "Not enough indices stored, call InitIndices with a higher N" << endl;
        cout << "Current reference i : " << i << " < " << ref_rho0.GetM() << endl;
        abort();
      }

      if (name_media == "M")
        ref_v0(i).SetInputData(nb, parameters, parameters(0));
      else if (name_media == "sigma")
        ref_sigma(i).SetInputData(nb, parameters, parameters(0));
      else if (name_media == "rho0")
        ref_rho0(i).SetInputData(nb, parameters, parameters(0));
      else if (name_media == "c0")
        ref_c0(i).SetInputData(nb, parameters, parameters(0));
      else if (name_media == "p0")
        ref_p0(i).SetInputData(nb, parameters, parameters(0));
      //    else if (name_media == "f0") //AJOUT NATHAN
      //      ref_f0(i).SetInputData(nb, parameters, parameters(0));
      else
      {
        cout << "Unknown media : " << name_media << endl;
        abort();
      }
    }


  //! returns the name associated with the physical index num
  template<class Dimension>
    string VarGalbrunIndex_Base<Dimension>::GetPhysicalIndexName(int m) const
    {
      switch(m)
      {
        case 0: return string("M");
        case 1: return string("sigma");
        case 2: return string("rho0");
        case 3: return string("c0");
        case 4: return string("p0");
        case 5: return string("f0");//AJOUT NATHAN
      }

      return string();
    }


  //! Are the physical indices variable inside domain ref ?
  template<class Dimension>
    bool VarGalbrunIndex_Base<Dimension>::IsVaryingMedia(int ref) const
    {
      if (ref_rho0(ref).IsVarying())
        return true;

      if (ref_c0(ref).IsVarying())
        return true;

      if (ref_p0(ref).IsVarying())
        return true;

      //    if (ref_f0(ref).IsVarying()) //AJOUT NATHAN
      //      return true;

      if (ref_v0(ref).IsVarying())
        return true;

      if (ref_sigma(ref).IsVarying())
        return true;

      return false;
    }


  //! returns the velocity of waves in a media
  template<class Dimension>
    Real_wp VarGalbrunIndex_Base<Dimension>::GetVelocityOfMedia(int ref) const
    {
      // we do not take into account the flow
      return this->ref_c0(ref).GetConstant();
    }


  //! returns the velocity of waves at infinity
  template<class Dimension>
    Real_wp VarGalbrunIndex_Base<Dimension>::GetVelocityOfInfinity() const
    {
      return 1.0;
    }


  //! fills the varying indices that need to be computed
  template<class Dimension>
    void VarGalbrunIndex_Base<Dimension>::
    GetVaryingIndices(Vector<PhysicalVaryingMedia<Dimension, Real_wp>* >& rho_real,
        IVect& num_ref, IVect& num_index, IVect& num_component,
        Vector<bool>& compute_grad, Vector<bool>& compute_hess)
    {
      int nb = 0;
      for (int i = 0; i < ref_v0.GetM(); i++)
      {
        nb += ref_v0(i).GetNbVaryingMedia();
        nb += ref_sigma(i).GetNbVaryingMedia();
        nb += ref_rho0(i).GetNbVaryingMedia();
        nb += ref_c0(i).GetNbVaryingMedia();
        nb += ref_p0(i).GetNbVaryingMedia();
        //        nb += ref_f0(i).GetNbVaryingMedia(); //AJOUT NATHAN
      }

      rho_real.Reallocate(nb);
      num_ref.Reallocate(nb);
      num_index.Reallocate(nb);
      num_component.Reallocate(nb);
      compute_grad.Reallocate(nb);
      compute_hess.Reallocate(nb);
      compute_grad.Fill(false);
      compute_hess.Fill(false);
      nb = 0;
      for (int i = 0; i < ref_v0.GetM(); i++)
      {
        int nb0 = nb;
        ref_v0(i).GetVaryingMedia(nb, rho_real, num_component);
        for (int j = nb0; j < nb; j++)
        {
          num_index(j) = 0;
          num_ref(j) = i;
          if ((compute_grad_flow) || (compute_div_flow))
            compute_grad(j) = true;

          if (compute_hessian_flow)
            compute_hess(j) = true;
        }

        //AJOUT NATHAN
        //        nb0 = nb;
        //        ref_f0(i).GetVaryingMedia(nb, rho_real, num_component);
        //        for (int j = nb0; j < nb; j++)
        //          {
        //            num_index(j) = 0;
        //            num_ref(j) = i;

        //          }


        nb0 = nb;
        ref_sigma(i).GetVaryingMedia(nb, rho_real, num_component);
        for (int j = nb0; j < nb; j++)
        {
          num_index(j) = 1;
          num_ref(j) = i;
          if (compute_grad_sigma)
            compute_grad(j) = true;
        }

        nb0 = nb;
        ref_rho0(i).GetVaryingMedia(nb, rho_real, num_component);
        for (int j = nb0; j < nb; j++)
        {
          num_index(j) = 2;
          num_ref(j) = i;
          if (compute_grad_rho || compute_grad_gamma)
            compute_grad(j) = true;
        }

        nb0 = nb;
        ref_c0(i).GetVaryingMedia(nb, rho_real, num_component);
        for (int j = nb0; j < nb; j++)
        {
          num_index(j) = 3;
          num_ref(j) = i;
          if (compute_grad_c0 || compute_grad_gamma)
            compute_grad(j) = true;
        }

        nb0 = nb;
        ref_p0(i).GetVaryingMedia(nb, rho_real, num_component);
        for (int j = nb0; j < nb; j++)
        {
          num_index(j) = 4;
          num_ref(j) = i;
          compute_grad(j) = true;
          if (compute_hessian_p0)
            compute_hess(j) = true;
        }
      }
    }  


  //! computes physical properties (flow, gradient and divergence of flow) on quadrature points  
  template<class Dimension>
    void VarGalbrunIndex_Base<Dimension>::ComputePhysicalCoefficients()
    {
      int nb_elt = var_problem.mesh.GetNbElt();
      eval_flow.Reallocate(nb_elt);
      eval_rho0.Reallocate(nb_elt);
      eval_rhoC2.Reallocate(nb_elt);
      eval_c0.Reallocate(nb_elt);
      eval_sigma.Reallocate(nb_elt);

      if(compute_source_momentum){
        eval_f0.Reallocate(nb_elt);
        if(compute_grad_momentum)
          grad_f0.Reallocate(nb_elt);
      }

      grad_p0.Reallocate(nb_elt);
      if (compute_grad_flow)
        grad_flow.Reallocate(nb_elt);

      if (compute_div_flow)
        div_flow.Reallocate(nb_elt);

      if (compute_grad_rho)
        grad_rho0.Reallocate(nb_elt);

      if (compute_grad_c0)
        grad_c0.Reallocate(nb_elt);

      if (compute_grad_sigma)
        grad_sigma.Reallocate(nb_elt);

      if (compute_hessian_flow)
        hessian_flow.Reallocate(nb_elt);

      if (compute_hessian_p0)
        hessian_p0.Reallocate(nb_elt);

      if (compute_gamma)
        eval_gamma.Reallocate(nb_elt);

      if (compute_grad_gamma)
        grad_gamma.Reallocate(nb_elt);

      R_N v0, nabla_rho0, nabla_c0; Real_wp c, p0, rho0;
      TinyMatrix<Real_wp, General, Dimension::dim_N, Dimension::dim_N> dv0;
      TinyVector<TinyMatrix<Real_wp, Symmetric, Dimension::dim_N, Dimension::dim_N>,
        Dimension::dim_N> hess_v0;

      for (int i = 0; i < nb_elt; i++)
      {
        int ref = var_problem.mesh.Element(i).GetReference();
        int N = var_problem.Glob_PointsQuadrature(i).GetM();
        eval_flow(i).Reallocate(N);
        eval_sigma(i).Reallocate(N);
        eval_rho0(i).Reallocate(N);
        eval_rhoC2(i).Reallocate(N);
        eval_c0(i).Reallocate(N);
        //AJOUT NATHAN
        if(compute_source_momentum)
        {
          eval_f0(i).Reallocate(N);
          if(compute_grad_momentum)
            grad_f0(i).Reallocate(N);
        }

        grad_p0(i).Reallocate(N);
        if (compute_grad_flow)
          grad_flow(i).Reallocate(N);

        if (compute_div_flow)
          div_flow(i).Reallocate(N);

        if (compute_grad_rho)
          grad_rho0(i).Reallocate(N);

        if (compute_grad_c0)
          grad_c0(i).Reallocate(N);

        if (compute_grad_sigma)
          grad_sigma(i).Reallocate(N);

        if (compute_hessian_flow)
          hessian_flow(i).Reallocate(N);

        if (compute_hessian_p0)
          hessian_p0(i).Reallocate(N);

        if (compute_gamma)
          eval_gamma(i).Reallocate(N);

        if (compute_grad_gamma)
          grad_gamma(i).Reallocate(N);

        for (int j = 0; j < N; j++)
        {
          if (compute_hessian_flow)
          {
            this->ref_v0(ref).GetCoefHessian(var_problem, i, j, v0, dv0, hess_v0);

            eval_flow(i)(j) = v0;	    
            grad_flow(i)(j) = dv0;
            hessian_flow(i)(j) = hess_v0;
          }
          else if ((compute_grad_flow) || (compute_div_flow))
          {
            this->ref_v0(ref).GetCoefGradient(var_problem, i, j, v0, dv0);

            eval_flow(i)(j) = v0;	    
            if (compute_grad_flow)
              grad_flow(i)(j) = dv0;

            if (compute_div_flow)
            {
              div_flow(i)(j) = 0;
              for (int k = 0; k < Dimension::dim_N; k++)
                div_flow(i)(j) += dv0(k, k);
            }
          }
          else
            eval_flow(i)(j) = this->ref_v0(ref).GetCoefficient(var_problem, i, j);

          if (compute_grad_rho || compute_grad_gamma)
            this->ref_rho0(ref).GetCoefGradient(var_problem, i, j, rho0, nabla_rho0);
          else
            rho0 = this->ref_rho0(ref).GetCoefficient(var_problem, i, j);

          eval_rho0(i)(j) = rho0;
          if (compute_grad_rho)
            grad_rho0(i)(j) = nabla_rho0;

          if (compute_grad_sigma)
            this->ref_sigma(ref).GetCoefGradient(var_problem, i, j, eval_sigma(i)(j), grad_sigma(i)(j));
          else
            eval_sigma(i)(j) = this->ref_sigma(ref).GetCoefficient(var_problem, i, j);

          if (compute_hessian_p0)
            this->ref_p0(ref).GetCoefHessian(var_problem, i, j, p0, grad_p0(i)(j), hessian_p0(i)(j));
          else
            this->ref_p0(ref).GetCoefGradient(var_problem, i, j, p0, grad_p0(i)(j));

          if (compute_grad_c0 || compute_grad_gamma)
            this->ref_c0(ref).GetCoefGradient(var_problem, i, j, c, nabla_c0);
          else
            c = this->ref_c0(ref).GetCoefficient(var_problem, i, j);

          if (compute_grad_c0)
            grad_c0(i)(j) = nabla_c0;

          eval_rhoC2(i)(j) = rho0*c*c;
          eval_c0(i)(j) = c;

          if (eval_gamma.GetM() > 0)
            eval_gamma(i)(j) = c*c*rho0/p0;

          if (grad_gamma.GetM() > 0)
            grad_gamma(i)(j) = 2.0*c*rho0/p0*nabla_c0 + c*c/p0*nabla_rho0 - c*c*rho0/(p0*p0)*grad_p0(i)(j);

          // storing grad( rho_0 c_0) or grad(c0^2)
          if (compute_grad_c0)
          {
            if (store_grad_rho0_c0)
              grad_c0(i)(j) = eval_rho0(i)(j)*grad_c0(i)(j) + c*grad_rho0(i)(j);
            else
              grad_c0(i)(j) = 2.0*c*grad_c0(i)(j);
          }

          //AJOUT NATHAN -- compute source f_0
          if(compute_source_momentum){    
            R_N f(0.0);
            Mlt(grad_flow(i)(j),eval_flow(i)(j),f);
            eval_f0(i)(j) = rho0*f+grad_p0(i)(j);

            //Compute grad momentum for Eulerian Galbrun
            if(compute_grad_momentum)
            {
              //Computing the Hess(v_0) part
              //TinyVector<TinyMatrix<Real_wp, Symmetric, Dimension::dim_N, Dimension::dim_N>, Dimension::dim_N > tmp_hess_v0; 
              //           TinyMatrix<Real_wp, General, Dimension::dim_N, Dimension::dim_N> tmp_hess_v0(0.0); 
              MatrixN_N tmp_hess_v0(0.0);
              MatrixN_N tmp_v(0.0), tmp_rho(0.0);
              for(int l=0; l<Dimension::dim_N; l++)
              {
                // COmputing rho_0*Hess(v_0)*v_0
                TinyMatrix<Real_wp, General, Dimension::dim_N, Dimension::dim_N> h0(0.0); 
                h0 = hessian_flow(i)(j)(l);
                TinyVector<Real_wp,Dimension::dim_N>  M = eval_flow(i)(j);
                TinyVector<Real_wp,Dimension::dim_N>  tmp;
                Mlt(h0,M,tmp);
                SetCol(tmp,l,tmp_hess_v0);

                //Preparing for tensor product ((\nabla v_0)v_0)\otimes \nabla\rho_0
                SetCol(f,l,tmp_v);
                SetRow(grad_rho0(i)(j),l,tmp_rho);
              }
              tmp_hess_v0 *= rho0;

              //Computing the \nabla v_0\nabla v_0 part
              TinyMatrix<Real_wp,General,Dimension::dim_N,Dimension::dim_N> grad_v_grad_v(0.0);
              Mlt(grad_flow(i)(j),grad_flow(i)(j),grad_v_grad_v);
              grad_v_grad_v *= rho0;

              //Computing the (\nabla v_0 v_0)\otimes \nabla\rho_0 part
              tmp_v = tmp_v*tmp_rho; //Component-wise multiplication

              //Computing \nabla f_0
              grad_f0(i)(j) = tmp_hess_v0+grad_v_grad_v+tmp_v+hessian_p0(i)(j);
            }
          }
          // checking Neumann condition on flow
          if (adjustment_neumann)
          {
            int nb_bounds = var_problem.mesh.Element(i).GetNbBoundary();
            int offset = var_problem.GetNbPointsQuadratureInside(i);
            for (int j = 0; j < nb_bounds; j++)
            {
              int num_face = var_problem.mesh.Element(i).numBoundary(j);
              int nb_pts_face = var_problem.mesh_num.GetNbPointsQuadratureBoundary(num_face);
              int ref = var_problem.mesh.Boundary(num_face).GetReference();
              if ((var_problem.mesh.GetBoundaryCondition(ref) == BoundaryConditionEnum::LINE_DIRICHLET) ||
                  (var_problem.mesh.GetBoundaryCondition(ref) == BoundaryConditionEnum::LINE_NEUMANN) )
              {
                for (int k = 0; k < nb_pts_face; k++)
                {
                  v0 = eval_flow(i)(offset + k);
                  ForceZeroVdotN(var_problem.Glob_normale(num_face)(k), v0);                        
                  eval_flow(i)(offset+k) = v0;
                }
              }

              offset += nb_pts_face;
            }
          }

          //DISP(i); DISP(j); DISP(eval_flow(i)(j));
          //Real_wp x = this->Glob_PointsQuadrature(i)(j)(0);
          //Real_wp y = this->Glob_PointsQuadrature(i)(j)(1);
          //Real_wp z = this->Glob_PointsQuadrature(i)(j)(2);
          /*Real_wp val_ex = -0.2*0.08*pi_wp/4*pi_wp/4*sin(pi_wp*x/4);
            if (abs(val_ex - hessian_flow(i)(j)(1)(0, 0)) > 1e-10)
            {
            cout << "Exact value is different : " << endl;
            DISP(val_ex); DISP(hessian_flow(i)(j)(1)(0, 0));
            abort();
            } */
        }
      }

      // stored values in ref_v0, etc are removed
      for (int ref = 0; ref < ref_v0.GetM(); ref++)
      {
        this->ref_v0(ref).Clear();
        this->ref_p0(ref).Clear();
        this->ref_c0(ref).Clear();
        this->ref_rho0(ref).Clear();
        this->ref_sigma(ref).Clear();
        //AJOUT NATHAN
        // this->ref_f0(ref).Clear();
      }

      //AJOUT NATHAN
      //Write f_0
      if(compute_source_momentum){
        int N = var_problem.mesh_num.GetNbDof();
        //DISP(N); DISP(var_problem.GetNbScalarDof());
        std::cout << "Writing source_f0...";
        VectReal_wp f(2*N), grad_f(4*N);//fx(N),fy(N);
        f.Zero();
        grad_f.Zero();
        for(int i=0; i<var_problem.mesh.GetNbElt(); i++){
          const VarProblem<Dimension,1>& var_pb = dynamic_cast<const VarProblem<Dimension,1>& >(var_problem);
          const ElementReference<Dimension,1>& Fb = var_pb.GetReferenceElement(i);
          IVect Nodle = var_problem.GetDofNumberOnElement(i);
          for(int j=0; j<Fb.GetNbDof(); j++){
            int numDof = Nodle(j);
            f(numDof) =  eval_f0(i)(j)(0);
            f(numDof+N) = eval_f0(i)(j)(1);

            if(compute_grad_momentum){
              grad_f(numDof) = grad_f0(i)(j)(0,0);
              grad_f(numDof+N) = grad_f0(i)(j)(0,1);
              grad_f(numDof+2*N) = grad_f0(i)(j)(1,0);
              grad_f(numDof+3*N) = grad_f0(i)(j)(1,1);
            }

          }
        }
        var_problem.GetOutputProblem().WriteOutputFile(f,"source_f",2);
        if(compute_grad_momentum)
          var_problem.GetOutputProblem().WriteOutputFile(grad_f,"grad_f",4);

        std::cout <<"Done" << std::endl;
      }
    }


  /*******************
   * VarGalbrun_Base *
   *******************/


  //! parameters of the data file, specific to Galbrun equation
  /*!
    \param[in] description_field keyword of the considered line of the data file
    \param[in] parameters list of values associated
    \param[in] nb_param number of values
    */
  void VarGalbrun_Base
    ::SetInputData(const string& description_field, const VectString& parameters)
    {
      if (description_field == "DropUnstableTerms")
      {
        if (parameters(0) == "Convective")
        {
          drop_unstable_terms = DROP_CONVECTIVE;
          if (parameters.GetM() > 1)
            coef_convective_term = to_num<Real_wp>(parameters(1));
          else
            coef_convective_term = 0.0;
        }
        else if (parameters(0) == "NonUniform")
          drop_unstable_terms = DROP_NON_UNIFORM;
        else
          drop_unstable_terms = DROP_NONE;
      }
      else if (description_field == "ApplyConvectiveCorrectionSource")
      {
        if (parameters(0) == "YES")
          apply_convective_derivate_source = true;
        else
          apply_convective_derivate_source = false;
      }
    }


  //! checks boundary conditions
  void VarGalbrun_Base
    ::CheckBoundaryCondition(const IVect& boundary_condition)
    {
      // for second-order formulation
      // an error is generated if the condition u.n = 0 is asked by the user
      if (!var_problem.FirstOrderFormulationDG())
      {
        for (int i = 0; i < boundary_condition.GetM(); i++)
        {
          int cond = boundary_condition(i);
          if (cond == BoundaryConditionEnum::LINE_DIRICHLET)
          {
            cout << "Condition u.n = 0 not implemented for this formulation" << endl;
            abort();
          }
        }
      }
    }

  /******************
   * VarGalbrun_Dim *
   ******************/


  template<class Complexe, class Dimension>
    void VarGalbrun_Dim<Complexe, Dimension>
    ::SetInputData(const string& description_field, const VectString& parameters)
    {
      VarGalbrun_Base::SetInputData(description_field, parameters);
      VarGalbrunIndex_Base<Dimension>::SetInputData(description_field, parameters);
    }


  //! applies (-i omega + sigma + M \cdot \nabla) to f if required
  template<class Complexe, class Dimension> template<class T0>
    void VarGalbrun_Dim<Complexe, Dimension>
    ::ModifyVolumetricSource(int i, int j, const typename Dimension::R_N& x,
        const VirtualSourceField<T0, Dimension>& fsrc,
        Vector<T0>& f) const
    {
      if (this->apply_convective_derivate_source)
      {
        Complexe m_iomega; var_problem.GetMiomega(m_iomega);
        Real_wp sigma = this->eval_sigma(i)(j);
        f *= m_iomega + sigma;

        Vector<T0> grad_f(f.GetM()*Dimension::dim_N);
        grad_f.Fill(0);
        fsrc.EvaluateGradient(x, grad_f);

        TinyVector<T0, Dimension::dim_N> grad_fn;
        Real_wp coef = this->coef_convective_term;
        for (int n = 0; n < f.GetM(); n++)
        {
          CopyVector(grad_f, n, grad_fn);
          f(n) += coef*DotProd(this->eval_flow(i)(j), grad_fn);
        }        
      }
    }


  /*******************
   * EllipticProblem *
   *******************/


  template<class Dimension>
    void EllipticProblem<HarmonicGalbrunEquation<Dimension> >
    ::AddElementaryFluxesDG(VirtualMatrix<Real_wp>& mat_sp,
        const GlobalGenericMatrix<Real_wp>& nat_mat,
        int offset_row, int offset_col)
    {
      cout << "Not possible" << endl;
      abort();
    }


  template<class Dimension>
    void EllipticProblem<HarmonicGalbrunEquation<Dimension> >
    ::AddElementaryFluxesDG(VirtualMatrix<Complex_wp>& mat_sp,
        const GlobalGenericMatrix<Complex_wp>& nat_mat,
        int offset_row, int offset_col)
    {
      Montjoie::AddElementaryFluxesDG(mat_sp, nat_mat, *this, offset_row, offset_col);
    }


  template<class Dimension>
    void EllipticProblem<HarmonicGalbrunEquation<Dimension> >
    ::ComputeElementaryMatrix(int iquad, IVect& num_dof, Matrix<Real_wp>& A,
        CondensationBlockSolver_Base<Real_wp>&,
        const GlobalGenericMatrix<Real_wp>& nat_mat)
    {
      cout << "Not possible" << endl;
      abort();
    }


  template<class Dimension>
    void EllipticProblem<HarmonicGalbrunEquation<Dimension> >
    ::ComputeElementaryMatrix(int iquad, IVect& num_dof, Matrix<Complex_wp>& A,
        CondensationBlockSolver_Base<Complex_wp>&,
        const GlobalGenericMatrix<Complex_wp>& nat_mat)
    {
      Montjoie::ComputeElementaryMatrix(iquad, num_dof, A, nat_mat, *this,
          this->GetReferenceElement(iquad));
    }


  template<class Dimension>
    void EllipticProblem<HarmonicGalbrunEquationDG<Dimension> >
    ::AddElementaryFluxesDG(VirtualMatrix<Real_wp>& mat_sp,
        const GlobalGenericMatrix<Real_wp>& nat_mat,
        int offset_row, int offset_col)
    {
      cout << "Not possible" << endl;
      abort();
    }


  template<class Dimension>
    void EllipticProblem<HarmonicGalbrunEquationDG<Dimension> >
    ::AddElementaryFluxesDG(VirtualMatrix<Complex_wp>& mat_sp,
        const GlobalGenericMatrix<Complex_wp>& nat_mat,
        int offset_row, int offset_col)
    {
      Montjoie::AddElementaryFluxesDG(mat_sp, nat_mat, *this, offset_row, offset_col);
    }


  template<class Dimension>
    void EllipticProblem<HarmonicGalbrunEquationDG<Dimension> >
    ::ComputeElementaryMatrix(int iquad, IVect& num_dof, Matrix<Real_wp>& A,
        CondensationBlockSolver_Base<Real_wp>&,
        const GlobalGenericMatrix<Real_wp>& nat_mat)
    {
      cout << "not possible" << endl;
      abort();
    }


  template<class Dimension>
    void EllipticProblem<HarmonicGalbrunEquationDG<Dimension> >
    ::ComputeElementaryMatrix(int iquad, IVect& num_dof, Matrix<Complex_wp>& A,
        CondensationBlockSolver_Base<Complex_wp>&,
        const GlobalGenericMatrix<Complex_wp>& nat_mat)
    {
      Montjoie::ComputeElementaryMatrix(iquad, num_dof, A, nat_mat, *this,
          this->GetReferenceElement(iquad));
    }


  template<class Dimension>
    void FemMatrixFreeClass<Real_wp, GalbrunStationaryEquationDG<Dimension> >
    ::MltAddFree(const GlobalGenericMatrix<Real_wp>& nat_mat,
        const SeldonTranspose& trans, int lvl, 
        const Vector<Real_wp>& X, Vector<Real_wp>& Y) const
    {
      MltAddVectorH1(Real_wp(1), nat_mat, trans, lvl, *this, X, Real_wp(1), Y, false);
    }


  template<class Dimension>
    void FemMatrixFreeClass<Real_wp, GalbrunStationaryEquationDG<Dimension> >
    ::MltAddFree(const GlobalGenericMatrix<Real_wp>& nat_mat,
        const SeldonTranspose& trans, int lvl, 
        const Vector<Complex_wp>& X, Vector<Complex_wp>& Y) const
    {
      cout << "not implemented" << endl;
      abort();
    }

}

#define MONTJOIE_FILE_HARMONIC_GALBRUN_CXX
#endif
