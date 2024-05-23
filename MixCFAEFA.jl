module MixCFAEFA


using Distributions, LinearAlgebra

export CFAmixEFA, boots_CFAmEFA_careless, boots_CFAmEFA_faking, classperf, inv_lemma_1_fun, inv_lemma_2_fun, det_lemma_1_fun, det_lemma_2_fun

#Y: data matrix in items x subjects form (p x n)
#q: CFA latent factors
#K: EFA latent factors
#maxIter: maximum number of rerun of the EM algorithm in case of non-convergence
#mu_hat: optional argument with 1 for estimation of CFA latent means, 0 for setting to zero the CFA means
#L_str: "Lambda structure" matrix of the same size of CFA factor loadings matrix (p x q), which indicates the items loading to a CFA latent factors by specifying ones in case of item-factor loading and zeros in absence of loading.
#       The matrix is needed if the items in Y are not partitioned in subsets loading to a single factor or if the number of items loading to different factors is heterogeneous. It automatically provides a Lambda structure matrix with no cross-loadings.

function CFAmixEFA(Y::Matrix{Float64}, q::Int64,  K::Int64, maxIter::Int64, X::Matrix{Float64}, maxNumRestart::Int64, mu_hat::Int64=1,L_str=nothing)
    local re::Int64, cal::Int64, non_conv::Bool, err2::Bool, sum_diff_tf::Bool, retr::Int64, res::Vector{Float64}, start_vals::Vector{Float64}, DomainError_warn::Int64, p::Int64, n::Int64
   
    
    ncov = size(X,2); p = size(Y,1); n = size(Y,2); iter_err = 1; code_error = 0.0;  err_0_count = 0; err_1_count = 0; err_2_count = 0; err_3_count = 0; err_4_count = 0; err_5_count = 0;
    while code_error != 8.0

        if iter_err >= maxNumRestart
            code_error = 7.0 
            return vcat(vec(fill(NaN,p,q)), vec(fill(NaN,p,K)), vec(diag(fill(NaN,p,p))), vec(diag(fill(NaN,p,p))), vec(fill(NaN,q,q)), vec(fill(NaN,n)), NaN, vec(fill(NaN,n)), vec(fill(NaN,ncov)), vec(fill(NaN,q)), vec(fill(NaN,K)), code_error, err_0_count, err_1_count, err_2_count, err_3_count, err_4_count, err_5_count) 
        end
        
        # Starting values
        start_vals = starting_val(n,p,q,K,X,mu_hat,L_str)

        # EM
        res = EM(Y,reshape(start_vals[((p*q)+(p*K)+p+p+q^2+n+ncov+p*q+1):((p*q)+(p*K)+p+p+q^2+n+ncov+p*q+n*ncov)],n,ncov),q,K,p,n,ncov,maxIter,reshape(start_vals[1:(p*q)],p,q), reshape(start_vals[(p*q+1):((p*q)+(p*K))],p,K), Diagonal(start_vals[((p*q)+(p*K)+1):((p*q)+(p*K)+p)]), Diagonal(start_vals[((p*q)+(p*K)+p+1):((p*q)+(p*K)+p+p)]), reshape(start_vals[((p*q)+(p*K)+p+p+1):((p*q)+(p*K)+p+p+q^2)],q,q), vec(start_vals[((p*q)+(p*K)+p+p+q^2+1):((p*q)+(p*K)+p+p+q^2+n)]), vec(start_vals[((p*q)+(p*K)+p+p+q^2+n+1):((p*q)+(p*K)+p+p+q^2+n+ncov)]), reshape(start_vals[((p*q)+(p*K)+p+p+q^2+n+ncov+1):((p*q)+(p*K)+p+p+q^2+n+ncov+p*q)],p,q),start_vals[((p*q)+(p*K)+p+p+q^2+n+ncov+p*q+n*ncov+1):((p*q)+(p*K)+p+p+q^2+n+ncov+p*q+n*ncov+q)],start_vals[((p*q)+(p*K)+p+p+q^2+n+ncov+p*q+n*ncov+q+1):((p*q)+(p*K)+p+p+q^2+n+ncov+p*q+n*ncov+q+K)],mu_hat)

        
        # Controls for non covnergence
        cal = (p*q)+(p*K)+p+p+q^2+n+1+n+ncov+q+K+1 
        sum_diff_tf = !(res[(cal+1):(cal+1)][1]==1.0)
        non_conv = res[(cal+1+1):(cal+1+1)][1]>=maxIter
        code_error = res[(cal):(cal)][1]
        if code_error == 0.0  && iter_err < maxNumRestart
            err_0_count += 1 
            iter_err += 1
        elseif code_error == 1.0  && iter_err < maxNumRestart
            err_1_count += 1 
            iter_err += 1
        elseif code_error == 2.0  && iter_err < maxNumRestart
            err_2_count += 1 
            iter_err += 1
        elseif code_error == 3.0 && iter_err < maxNumRestart
            err_3_count += 1 
            iter_err += 1
        elseif non_conv && iter_err < maxNumRestart
            err_4_count += 1 
            iter_err += 1
        elseif sum_diff_tf && iter_err < maxNumRestart
            err_5_count += 1 
            iter_err += 1
        end
    end 
    return res[1:(cal-1)]
end

# Expectation-Maximization algorithm
function EM(Y::Matrix{Float64},X::Matrix{Float64},q::Int64, K::Int64, p::Int64, n::Int64, ncov::Int64, maxIter::Int64, Lambda_1::Matrix{Float64}, Lambda_2::Matrix{Float64}, Theta_delta::Diagonal{Float64, Vector{Float64}}, Psi_delta::Diagonal{Float64, Vector{Float64}}, Phi::Matrix{Float64}, kappa::Vector{Float64}, betas::Vector, L_str::Matrix, mu::Vector{Float64}, nu::Vector{Float64}, mu_hat::Int64)
    local llik::Vector{Float64}, Ez1::Vector{Float64}, diff_llik::Float64, oldllik::Float64, dtol::Float64, t::Int64, err::Vector{Bool}, err2::Bool, sum_diff_tf::Bool, sum_diff::Vector{Bool}, old_diff::Float64,  ESxixi::Matrix{Float64}, ESetaeta::Matrix{Float64}, ESyxi::Matrix{Float64}, ESyeta::Matrix{Float64}, ESyy0::Matrix{Float64}, ESyy1::Matrix{Float64}, Ek0::Float64, Ek1::Float64, Ez0::Vector{Float64}, B2::Matrix{Float64}, b2::Matrix{Float64}, B1::Matrix{Float64}, b1::Matrix{Float64}

            
    t = 1; err2 = false; llik = zeros(Float64, maxIter); oldllik = -Inf; err=zeros(Bool,maxIter); sum_diff = zeros(Bool,maxIter); sum_diff_tf = true;  diff_llik = 1.0; dtol = 0.0005; code_error = 8.0;
    while diff_llik>dtol && t<=maxIter && sum_diff_tf
            inv_lemma_1 = inv_lemma_1_fun(q,Lambda_1,Theta_delta,Phi)
            inv_lemma_2 = inv_lemma_2_fun(K,Lambda_2,Psi_delta)      
            det_lemma_1 = det_lemma_1_fun(Lambda_1,Theta_delta,Phi)
            det_lemma_2 = det_lemma_2_fun(K,Lambda_2,Psi_delta)

            if det_lemma_1<=0 || det_lemma_2<=0
                code_error = 0.0
                return vcat(vec(fill(NaN,p,q)), vec(fill(NaN,p,K)), vec(diag(fill(NaN,p,p))), vec(diag(fill(NaN,p,p))), vec(fill(NaN,q,q)), vec(fill(NaN,n)), NaN, vec(fill(NaN,n)), vec(fill(NaN,ncov)), vec(fill(NaN,q)), vec(fill(NaN,K)), code_error, sum_diff_tf, t-1) 
            end
            println(t)
            #E-step
            # b1 = b1_fun(q,Lambda_1,Phi,Theta_delta,mu)
            # B1 = B1_fun(Lambda_1,Phi,Theta_delta,b1,mu)
            # b2 = b2_fun(K,Lambda_2,Psi_delta,nu)
            # B2 = B2_fun(K,Lambda_2,Psi_delta,b2,nu)
            
            b1 = b1_fun(q,Lambda_1,Phi,Theta_delta)
            B1 = B1_fun(Lambda_1,Phi,Theta_delta,b1)
            b2 = b2_fun(K,Lambda_2,Psi_delta)
            B2 = B2_fun(K,Lambda_2,Psi_delta,b2)
            
            Ez1 = Expectz(n,p,Y,kappa,inv_lemma_1,inv_lemma_2, det_lemma_1,det_lemma_2,Lambda_1,Lambda_2,mu,nu)

            Ez0 = Expectz0(Ez1)
            Ek1 = Ekz1(Ez1)
            Ek0 = Ekz0(Ez0)
            
            ESyy1 = ESyyz1(n,p,Y,Ez1)  
            ESyy0 = ESyyz0(n,p,Y,Ez0)
            
            Ezy1 = ESzy1(n,p,Y,Ez1,mu,Lambda_1)
            Ezy0 = ESzy0(n,p,Y,Ez0,nu,Lambda_2)
            Ey1 = ESyone(n,p,Y,Ez1)
            Ey0 = ESyzero(n,p,Y,Ez0)
            
            Eeta = Eeta1(p,n,Ek1,Ez1,Y,b1, mu, Lambda_1)
            Exi = Exi0(p,n,Ek0,Ez0,Y,b2, nu, Lambda_2)
            
            Eetay1 =  ESetay1(n,p,Y,Ez1,Lambda_1,mu,b1,Ey1)
            Exiy0 =  ESxiy0(n,p,Y,Ez0,Lambda_2,nu,b2,Ey0)

            ESetaeta =  ESetaeta1(n,p,Y,Ez1,Lambda_1,mu,b1,B1)
            ESxixi = ESxixi0(n,p,Y,Ez0,Lambda_2,nu,b2,B2)
            if isequal(sum(ESetaeta),NaN) || isequal(sum(Eetay1),NaN) || isnan(ESxixi[1,1]) || isequal(sum(ESxixi),0)
                code_error = 0.0
                return vcat(vec(fill(NaN,p,q)), vec(fill(NaN,p,K)), vec(diag(fill(NaN,p,p))), vec(diag(fill(NaN,p,p))), vec(fill(NaN,q,q)), vec(fill(NaN,n)), NaN, vec(fill(NaN,n)), vec(fill(NaN,ncov)), vec(fill(NaN,q)), vec(fill(NaN,K)), code_error, sum_diff_tf, t-1) 
            else
        
                
            #M-step
            Lambda_1 = Lambda_1_mstep(Eetay1,ESetaeta,L_str)
            Lambda_2 = Lambda_2_mstep(Exiy0,ESxixi)
            
            Theta_delta = Theta_delta_mstep(n,Lambda_1,ESyy1,Eetay1,Ek1)
            Psi_delta = Psi_delta_mstep(n,Lambda_2,ESyy0,Exiy0,Ek0)#Psi_delta_mstep(Y,b2,B2,nu,n,p,K,Lambda_2,ESyy0,Exiy0,Ez0)
             

            if mu_hat == 1
                mu = ((sum(Eeta,dims=2)/n)/(Ek1/n))[1:q,1]
            elseif mu_hat == 0
                mu = zeros(q)
            end
            nu = ((sum(Exi,dims=2)/n)/(Ek0/n))[1:K,1]
            
            Phi = Phi_mstep(ESetaeta,Eeta,Ek1,mu,n)
          
            #If no covariates are provided
            if ncov == 1
                kappa_1 = Ek1/n
                kappa = repeat([kappa_1],n)
                betas = log(kappa[1]/(1 - kappa[1]))
            else
                betas = NewtonRaphson(betas_gradient, betas, X, Ez1, Ez0, ncov)
                kappa = exp.(X*betas)./(exp.(X*betas).+1)
            end         
           
            #Controls for non-convergence
            if sum(Theta_delta.<0) > 0 || det(Psi_delta) < 0 || isequal(sum(Theta_delta),NaN) || isequal(sum(Ez0),NaN)
                code_error = 0.0
                return vcat(vec(fill(NaN,p,q)), vec(fill(NaN,p,K)), vec(diag(fill(NaN,p,p))), vec(diag(fill(NaN,p,p))), vec(fill(NaN,q,q)), vec(fill(NaN,n)), NaN, vec(fill(NaN,n)), vec(fill(NaN,ncov)), vec(fill(NaN,q)), vec(fill(NaN,K)), code_error, sum_diff_tf, t-1) 
            elseif isnan(betas[1]) 
                code_error = 1.0
                return vcat(vec(fill(NaN,p,q)), vec(fill(NaN,p,K)), vec(diag(fill(NaN,p,p))), vec(diag(fill(NaN,p,p))), vec(fill(NaN,q,q)), vec(fill(NaN,n)), NaN, vec(fill(NaN,n)), vec(fill(NaN,ncov)), vec(fill(NaN,q)), vec(fill(NaN,K)), code_error, sum_diff_tf, t-1) 
            elseif det(Phi)<0 
                code_error = 2.0
                return vcat(vec(fill(NaN,p,q)), vec(fill(NaN,p,K)), vec(diag(fill(NaN,p,p))), vec(diag(fill(NaN,p,p))), vec(fill(NaN,q,q)), vec(fill(NaN,n)), NaN, vec(fill(NaN,n)), vec(fill(NaN,ncov)), vec(fill(NaN,q)), vec(fill(NaN,K)), code_error, sum_diff_tf, t-1) 
            else
                old_diff = diff_llik
                llik[t] = llik_compute(Y,n,p,q,K,Lambda_1, Lambda_2, Theta_delta, Psi_delta, Phi, kappa, Ez1, Ez0,Ek0,Ek1,mu,nu,Eeta,Exi,Eetay1,Exiy0,ESyy1,ESyy0,ESetaeta,ESxixi)
 
                diff_llik = abs(oldllik - llik[t])
                if !(oldllik<llik[t])
                    code_error = 3.0
                    return vcat(vec(fill(NaN,p,q)), vec(fill(NaN,p,K)), vec(diag(fill(NaN,p,p))), vec(diag(fill(NaN,p,p))), vec(fill(NaN,q,q)), vec(fill(NaN,n)), NaN, vec(fill(NaN,n)), vec(fill(NaN,ncov)), vec(fill(NaN,q)), vec(fill(NaN,K)), code_error, sum_diff_tf, t-1) 
                end

                sum_diff[t] = diff_llik>old_diff               
                oldllik = llik[t]
                t=t+1;
                sum_diff_tf = sum(sum_diff) < 150            
            end
        end
        end
    return vcat(vec(Lambda_1), vec(Lambda_2), vec(diag(Theta_delta)), vec(diag(Psi_delta)), vec(Phi), kappa, llik[t-1], Ez1, betas, mu, nu, code_error, sum_diff_tf, t-1);
end        

#Beta gradient
function betas_gradient(betas::Vector,X::Matrix{Float64},Ez1::Vector{Float64},Ez0::Vector{Float64})
    fval = - X'*((-Ez1.+Ez0.*exp.(X*betas))./(exp.(X*betas).+1))
    return vec(fval)
end



#Newton-Raphson
function NewtonRaphson(betas_gradient, betas::Vector, X::Matrix{Float64}, Ez1::Vector{Float64}, Ez0::Vector{Float64}, ncov::Int64)
    local betas_old::Vector{Float64}, iter::Int64, tol::Float64, maxIter::Int64, grad::Vector{Float64}, hess::Matrix{Float64}
    betas_old = repeat([-Inf],size(X,2)); iter = 0;  tol=1e-8; maxIter = 1000
    while sum(abs.(betas.-betas_old)) > tol && iter < maxIter
        grad = betas_gradient(betas,X,Ez1,Ez0)
        hess = -(X.*exp.(X*betas)./(exp.(X*betas).+1).^2)'*X
        # println(hess)
        if det(hess) == 0.0 || isnan(sum(hess)) || isinf(sum(hess))
            return vec(fill(NaN,ncov))
        else
        betas_old = betas
        betas = betas - hess\grad
        iter += 1
        end
        if iter >= maxIter
            betas = vec(fill(NaN,ncov))
        end
    end
    return betas
end



# Starting values
function starting_val(n::Int64, p::Int64, q::Int64, K::Int64, X::Matrix, mu_hat::Int64,L_str=nothing)
    local Lambda_1::Matrix{Float64}, Lambda_2::Matrix{Float64}, Theta_delta::Diagonal{Float64, Vector{Float64}}, Psi_delta::Diagonal{Float64, Vector{Float64}}, Phi::Matrix{Float64}, kappa::Vector{Float64}

    #CFA
    if L_str === nothing
        L_str = L_str_fun(p,q)
    end
    Lambda_1 = rand(Uniform(0,1),p,q) .* L_str
    Phi = rand(LKJ(q,1.0))
    Theta_delta = Diagonal(vec(1 .- sum(Lambda_1,dims=2).^2));
    if mu_hat == 1
        mu = rand(Uniform(-2,2),q) 
    elseif mu_hat == 0
        mu = zeros(q)
    end
        
    #EFA
    Lambda_2 = rand(Uniform(0,1),p,K)
    Psi_delta = Diagonal(rand(Uniform(0.05,2),p))
    nu = rand(Uniform(-2,4),K) 

    #Mixture    
    betas = rand(Uniform(-1.5,1.5), size(X,2))
    kappa = exp.(X*betas)./(exp.(X*betas).+1)

    return vcat(vec(Lambda_1), vec(Lambda_2), vec(diag(Theta_delta)), vec(diag(Psi_delta)), vec(Phi), kappa, betas, vec(L_str), vec(X), mu, nu)
end

# Construction of Lambda structure matrix
function L_str_fun(p::Int64,q::Int64)
    local A::Matrix{Int64}, str::Vector{Int64}, L_str::Matrix{Float64}
    A =  Matrix{Int64}(I,q,q)
    str = ones(Int64,Int64(p/q))
    L_str = kron(A, str);
    return L_str
end

# More convenient matrix calculus
function inv_lemma_1_fun(q::Int64,Lambda_1::Matrix{Float64},Theta_delta::Diagonal{Float64, Vector{Float64}},Phi::Matrix{Float64})
    local inv_lemma::Matrix{Float64}
    inv_lemma = (inv(Theta_delta)-(Theta_delta\Lambda_1)*((Matrix{Float64}(I,q,q)+Phi*Lambda_1'*(Theta_delta\Lambda_1))\Phi)*Lambda_1'*inv(Theta_delta))
    return inv_lemma
end

function inv_lemma_2_fun(K::Int64,Lambda_2::Matrix{Float64},Psi_delta::Diagonal{Float64, Vector{Float64}})
    local inv_lemma::Matrix{Float64}
    inv_lemma = (inv(Psi_delta)-(Psi_delta\Lambda_2)*((Matrix{Float64}(I,K,K)+Lambda_2'*(Psi_delta\Lambda_2))\Lambda_2')*inv(Psi_delta))
    return inv_lemma
end

function det_lemma_1_fun(Lambda_1::Matrix{Float64},Theta_delta::Diagonal{Float64, Vector{Float64}},Phi::Matrix{Float64})
    local det_lemma::Float64
    det_lemma = det(inv(Phi)+Lambda_1'*(Theta_delta\Lambda_1))*det(Phi)*det(Theta_delta)
    return det_lemma
end

function det_lemma_2_fun(K::Int64,Lambda_2::Matrix{Float64},Psi_delta::Diagonal{Float64, Vector{Float64}})
    local det_lemma::Float64
    det_lemma = det(Matrix{Float64}(I,K,K)+Lambda_2'*(Psi_delta\Lambda_2))*det(Psi_delta)
    return det_lemma
end



# Functions for E-step

function b1_fun(q::Int64,Lambda_1::Matrix{Float64},Phi::Matrix{Float64},Theta_delta::Diagonal{Float64, Vector{Float64}})
    local b1::Matrix{Float64}
    b1 = Phi*Lambda_1'*inv(Theta_delta)-Phi*Lambda_1'*(Theta_delta\Lambda_1)*((Matrix{Float64}(I,q,q)+Phi*Lambda_1'*(Theta_delta\Lambda_1))\Phi)*Lambda_1'*inv(Theta_delta)
    return b1
end

function B1_fun(Lambda_1::Matrix{Float64},Phi::Matrix{Float64},Theta_delta::Diagonal{Float64, Vector{Float64}}, b1::Matrix{Float64})
    local B1::Matrix{Float64}
    B1 = Phi-b1*Lambda_1*Phi
    return B1
end

function b2_fun(K::Int64,Lambda_2::Matrix{Float64},Psi_delta::Diagonal{Float64, Vector{Float64}})
    local b2::Matrix{Float64}
    b2 =  Lambda_2'*inv(Psi_delta)-Lambda_2'*(Psi_delta\Lambda_2)*((Matrix{Float64}(I,K,K)+Lambda_2'*(Psi_delta\Lambda_2))\Lambda_2')*inv(Psi_delta)
    return b2
end

function B2_fun(K::Int64,Lambda_2::Matrix{Float64},Psi_delta::Diagonal{Float64, Vector{Float64}}, b2::Matrix{Float64})
    local B2::Matrix{Float64}
    B2 = Matrix{Float64}(I,K,K)-b2*Lambda_2
    return B2
end


function Expectz(n::Int64, p::Int64, Y::Matrix{Float64},kappa::Vector{Float64},inv_lemma_1::Matrix{Float64},inv_lemma_2::Matrix{Float64}, det_lemma_1::Float64,det_lemma_2::Float64, Lambda_1::Matrix{Float64}, Lambda_2::Matrix{Float64}, mu::Vector{Float64}, nu::Vector{Float64})
    local Ez1::Vector{Float64}
    
    Ez1 = zeros(Float64,n)
    @inbounds  for i in 1:n
        Ez1[i] = ((kappa[i] * ((exp((-1/2) * (Y[:,i]-Lambda_1*mu)'*inv_lemma_1*(Y[:,i]-Lambda_1*mu))/(sqrt((2 * pi)^(p) *det_lemma_1)))))/
        ((kappa[i] * ((exp((-1/2) * (Y[:,i]-Lambda_1*mu)'*inv_lemma_1*(Y[:,i]-Lambda_1*mu))/(sqrt((2 * pi)^(p) *det_lemma_1)))) 
        +((exp((-1/2) * (Y[:,i]-Lambda_2*nu)'*inv_lemma_2*(Y[:,i]-Lambda_2*nu)))/(sqrt((2 * pi)^(p) *det_lemma_2))) 
        -(kappa[i] *((exp((-1/2) * (Y[:,i]-Lambda_2*nu)'*inv_lemma_2*(Y[:,i]-Lambda_2*nu)))/(sqrt((2 * pi)^(p) *det_lemma_2))))))) 
    end
    return Ez1
end

function Expectz0(Ez1::Vector{Float64})
    local Ez0::Vector{Float64}
    Ez0 = @. 1 - Ez1
    return Ez0
end

function Ekz1(Ez1::Vector{Float64})
    local Ek1::Float64
    Ek1 = sum(Ez1)
    return Ek1
end

function Ekz0(Ez0::Vector{Float64})
    local Ek0::Float64
    Ek0 = sum(Ez0)
    return Ek0
end

function ESyyz1(n::Int64,p::Int64,Y::Matrix{Float64},Ez1::Vector{Float64})
    local ESyy1::Matrix{Float64}

    ESyy1 = (Y.*Ez1')*Y'/n   
end


function ESyyz0(n::Int64,p::Int64,Y::Matrix{Float64},Ez0::Vector{Float64})
    local ESyy0::Matrix{Float64}

    ESyy0 = (Y.*Ez0')*Y'/n  
end

function ESzy1(n::Int64,p::Int64,Y::Matrix{Float64},Ez1::Vector{Float64},mu::Vector{Float64},Lambda_1::Matrix{Float64})
    local Ezy1::Matrix{Float64}

    Ezy1 = sum((Y.-Lambda_1*mu).*Ez1',dims=2)/n     
end

function ESzy0(n::Int64,p::Int64,Y::Matrix{Float64},Ez0::Vector{Float64},nu::Vector{Float64},Lambda_2::Matrix{Float64})
    local Ezy0::Matrix{Float64}

    Ezy0 = sum((Y.-Lambda_2*nu).*Ez0',dims=2)/n
end

function ESyone(n::Int64,p::Int64,Y::Matrix{Float64},Ez1::Vector{Float64})
    local Ey1::Matrix{Float64}

    Ey1 = sum(Y.*Ez1',dims=2)/n     
end

function ESyzero(n::Int64,p::Int64,Y::Matrix{Float64},Ez0::Vector{Float64})
    local Ey0::Matrix{Float64}

    Ey0 = sum(Y.*Ez0',dims=2)/n
end

function Eeta1(p::Int64,n::Int64,Ek1::Float64,Ez1::Vector{Float64},Y::Matrix{Float64},b1::Matrix{Float64}, mu::Vector{Float64}, Lambda_1::Matrix{Float64})
    local Eeta::Matrix{Float64}
   
    Eeta = b1*(Ez1'.*(Y.-Lambda_1*mu)) + mu*Ez1'   
    return Eeta
end

function Exi0(p::Int64,n::Int64,Ek0::Float64,Ez0::Vector{Float64},Y::Matrix{Float64},b2::Matrix{Float64}, nu::Vector{Float64}, Lambda_2::Matrix{Float64})
    local Exi::Matrix{Float64}

    Exi =  b2*(Ez0'.*(Y.-Lambda_2*nu)) + nu*Ez0' 
    return Exi
end

function ESetay1(n::Int64,p::Int64,Y::Matrix{Float64},Ez1::Vector{Float64},Lambda_1::Matrix{Float64},mu::Vector{Float64},b1::Matrix{Float64},Ey1::Matrix{Float64})
    local Eetay1::Matrix{Float64}

    Eetay1 = (b1*((Ez1'.*(Y.-Lambda_1*mu)) *Y'))'/n .+ Ey1*mu' 
end

function ESxiy0(n::Int64,p::Int64,Y::Matrix{Float64},Ez0::Vector{Float64},Lambda_2::Matrix{Float64},nu::Vector{Float64},b2::Matrix{Float64},Ey0::Matrix{Float64})
    local Exiy0::Matrix{Float64}

    Exiy0 = (b2*((Ez0'.*(Y.-Lambda_2*nu)) *Y'))'/n .+ Ey0*nu' 
end

function ESetaeta1(n::Int64,p::Int64,Y::Matrix{Float64},Ez1::Vector{Float64},Lambda_1::Matrix{Float64},mu::Vector{Float64},b1::Matrix{Float64},B1::Matrix{Float64})
    local ESetaeta::Matrix{Float64}

    ESetaeta =  B1*(sum(Ez1)/n) + (b1*(((Y.-Lambda_1*mu).*Ez1')*(Y.-Lambda_1*mu)')/n)*b1' + mu*mu'*(sum(Ez1)/n) + mu*sum((Ez1.*(Y.-Lambda_1*mu)'),dims=1)/n*b1'+b1*(sum((Ez1.*(Y.-Lambda_1*mu)'),dims=1)/n)'*mu'
end

function ESxixi0(n::Int64,p::Int64,Y::Matrix{Float64},Ez0::Vector{Float64},Lambda_2::Matrix{Float64},nu::Vector{Float64},b2::Matrix{Float64},B2::Matrix{Float64})
    local ESxixi::Matrix{Float64}

    ESxixi =  B2*(sum(Ez0)/n) + (b2*(((Y.-Lambda_2*nu).*Ez0')*(Y.-Lambda_2*nu)')/n)*b2' + nu*nu'*(sum(Ez0)/n) + nu*sum((Ez0.*(Y.-Lambda_2*nu)'),dims=1)/n*b2'+b2*(sum((Ez0.*(Y.-Lambda_2*nu)'),dims=1)/n)'*nu'
end

# (Matrix{Float64}(I,K,K)-nu*nu'+(Lambda_2'-nu*nu'*Lambda_2')*inv(Lambda_2*Lambda_2'+Psi_delta-Lambda_2*nu*nu'*Lambda_2')*(Lambda_2-Lambda_2*nu*nu')*(Ek0/n)+(nu.+(Lambda_2'-nu*nu'Lambda_2')*inv(Lambda_2*Lambda_2'+Psi_delta-Lambda_2*nu*nu'*Lambda_2')*(Y-Lambda_2*nu))*(nu.+(Lambda_2'-nu*nu'Lambda_2')*inv(Lambda_2*Lambda_2'+Psi_delta-Lambda_2*nu*nu'*Lambda_2')*(Y-Lambda_2*nu))')

# Functions for M-step

function Lambda_1_mstep(Eetay1::Matrix{Float64},ESetaeta::Matrix{Float64},L_str::Matrix)
    local Lambda_1::Matrix{Float64}
    Lambda_1 = Eetay1*inv(ESetaeta) .* L_str
    return Lambda_1
end

function Lambda_2_mstep(Exiy0::Matrix{Float64},ESxixi::Matrix{Float64})
    local Lambda_2::Matrix{Float64}
    Lambda_2 = Exiy0*inv(ESxixi) 
    return Lambda_2
end

function Theta_delta_mstep(n::Int64,Lambda_1::Matrix{Float64},ESyy1::Matrix{Float64},Eetay1::Matrix{Float64},Ek1::Float64)
    local Theta_delta::Diagonal{Float64, Vector{Float64}}
    Theta_delta = Diagonal((ESyy1 - Lambda_1*Eetay1')/(Ek1/n))
    return Theta_delta
end

function Psi_delta_mstep(n::Int64,Lambda_2::Matrix{Float64},ESyy0::Matrix{Float64},Exiy0::Matrix{Float64},Ek0::Float64)
    local Psi_delta::Diagonal{Float64, Vector{Float64}}
    Psi_delta = Diagonal((ESyy0 - Lambda_2*Exiy0')/(Ek0/n))
    return Psi_delta
end




function Phi_mstep(ESetaeta::Matrix{Float64},Eeta::Matrix{Float64},Ek1::Float64,mu::Vector{Float64},n::Int64)
    local Phi::Matrix{Float64}
    Phi = (ESetaeta-sum(Eeta,dims=2)*mu'/n)/(Ek1/n)
    Phi = sqrt(inv(Diagonal(Phi)))*Phi*sqrt(inv(Diagonal(Phi)))
    return Phi
end



# Log-likelihood computation

function llik_compute(Y::Matrix{Float64},n::Int64,p::Int64,q::Int64,K::Int64,Lambda_1::Matrix{Float64}, Lambda_2::Matrix{Float64}, Theta_delta::Diagonal{Float64, Vector{Float64}}, Psi_delta::Diagonal{Float64, Vector{Float64}}, Phi::Matrix{Float64}, kappa::Vector{Float64}, Ez1::Vector{Float64}, Ez0::Vector{Float64},Ek0::Float64,Ek1::Float64,mu::Vector{Float64},nu::Vector{Float64},Eeta::Matrix{Float64},Exi::Matrix{Float64},Eetay1::Matrix{Float64},Exiy0::Matrix{Float64},ESyy1::Matrix{Float64},ESyy0::Matrix{Float64},ESetaeta::Matrix{Float64},ESxixi::Matrix{Float64},)
    local llik::Float64, transpY::Adjoint{Float64, Matrix{Float64}}, tr1::Float64, tr2::Float64, sum1::Float64, tr3::Float64, sum2::Float64, tr4::Float64, tr5::Float64, tr6::Float64, sum3::Float64, tr7::Float64, sum4::Float64, tr8::Float64, B1_intra::Matrix{Float64}, B2_intra::Matrix{Float64}, b1_intra::Matrix{Float64}, b2_intra::Matrix{Float64}
    transpY = Adjoint{Float64, Matrix{Float64}}(Y)
    
    tr1 = tr(Theta_delta\ESyy1)*n
    tr2 = 2*tr(Lambda_1'*(Theta_delta\Eetay1))*n
    tr3 = tr(Lambda_1'*(Theta_delta\Lambda_1)*ESetaeta)*n
    tr4 = tr(Phi\ESetaeta)*n
    tr5 = 2*tr(inv(Phi)*sum(Eeta,dims=2)*mu')
    tr6 = tr((Phi)\(Ek1*mu*mu'))
    
    tr7 = tr(Psi_delta\ESyy0)*n
    tr8 = 2*tr(Lambda_2'*(Psi_delta\Exiy0))*n
    tr9 = tr(Lambda_2'*(Psi_delta\Lambda_2)*ESxixi)*n
    tr10 = tr(ESxixi)*n
    tr11 = 2*tr(sum(Exi,dims=2)*nu')
    tr12 = tr((Ek0*nu*nu'))

    lk1 = sum(Ez1.*log.(kappa))
    lk0 = sum(Ez0.*log.(1 .-kappa))
    lT = sum(Ez1)*(1/2)*log(det(Theta_delta))
    lPh = sum(Ez1)*(1/2)*log(det(Phi))
    lPs = sum(Ez0)*(1/2)*log(det(Psi_delta))

    llik = (lk1-sum(Ez1)*(p/2)*log(2*pi)-lT-((1/2)*(tr1-tr2+tr3))-sum(Ez1)*(q/2)*log(2*pi)-lPh-(1/2)*(tr4-tr5+tr6))+
    (lk0-sum(Ez0)*(p/2)*log(2*pi)-lPs-((1/2)*(tr7-tr8+tr9))-sum(Ez0)*(K/2)*log(2*pi)-(1/2)*(tr10-tr11+tr12))
    return llik
end


function classperf(z,Ez)
    TrueNegatives = sum(Ez.== 1 .&& z.==  1)
    FalseNegatives = sum(Ez.!= 1 .&& z .== 1)
    FalsePositives = sum(Ez.== 1 .&& z .!= 1)
    TruePositives = sum(Ez.!= 1 .&& z .!= 1)


    SP = TrueNegatives/(TrueNegatives+FalsePositives)
    SE = TruePositives/(TruePositives+FalseNegatives)
    MCC = (TruePositives*TrueNegatives-FalsePositives*FalseNegatives)/sqrt((TruePositives+FalsePositives)*(TruePositives+FalseNegatives)*(TrueNegatives+FalsePositives)*(TrueNegatives+FalseNegatives))
    BACC = (TruePositives/(TruePositives+FalseNegatives)+(TrueNegatives/(TrueNegatives+FalsePositives))) / 2

    return vcat(hcat("BACC","MCC","SE","SP"),hcat(BACC,MCC,SE,SP))
end


function boots_CFAmEFA_careless(Y,X,n,p,q,K,B,std_predictors,L_str,mu,str=missing)
    std_preds = vcat(1,std_predictors)
    ncov = size(X,2); Lambda_1_boots = Array{Float64}(undef,p,B); Theta_delta_boots = Array{Float64}(undef,p,B); Phi_boots = Array{Float64}(undef,q*q,B); Lambda_2_boots = Array{Float64}(undef,p*K,B); Psi_delta_boots = Array{Float64}(undef,p,B); beta_boots = Array{Float64}(undef,ncov,B); mu_boots = Array{Float64}(undef,q,B); nu_boots = Array{Float64}(undef,K,B); Ez1_boots = Array{Float64}(undef,n,B);
    for b in 1:B
        println(b)
        boots = sample(1:708,708,replace=true)
        
        
        Y = Y[:,boots]
        X = X[boots,:]
        
        res =  CFAmixEFA(Y, q, K, 500, X, 2000, mu, L_str)
      

        Lambda_1_boots[:,b] = sum(reshape(res[1:(p*q)],p,q),dims=2)
        Lambda_2_boots[:,b] = res[(p*q+1):((p*q)+(p*K))]
        Theta_delta_boots[:,b] = res[((p*q)+(p*K)+1):((p*q)+(p*K)+p)]
        Psi_delta_boots[:,b] = res[((p*q)+(p*K)+p+1):((p*q)+(p*K)+p+p)]
        Phi_boots[:,b] = res[((p*q)+(p*K)+p+p+1):((p*q)+(p*K)+p+p+q^2)]
        Ez1_boots[:,b] = res[((p*q)+(p*K)+p+p+q^2+n+1+1):((p*q)+(p*K)+p+p+q^2+n+1+n)]
        beta_boots[:,b] = res[((p*q)+(p*K)+p+p+q^2+n+1+n+1):((p*q)+(p*K)+p+p+q^2+n+1+n+ncov)]./std_preds
        mu_boots[:,b] = res[((p*q)+(p*K)+p+p+q^2+n+1+n+ncov+1):((p*q)+(p*K)+p+p+q^2+n+1+n+ncov+q)]
        nu_boots[:,b] = res[((p*q)+(p*K)+p+p+q^2+n+1+n+ncov+q+1):((p*q)+(p*K)+p+p+q^2+n+1+n+ncov+q+K)]

        boo = vcat(Lambda_1_boots[:,b], Theta_delta_boots[:,b], Phi_boots[:,b], Lambda_2_boots[:,b], Psi_delta_boots[:,b], beta_boots[:,b], mu_boots[:,b], nu_boots[:,b])
        if str !== missing
            jldsave(string(str,"/boots_",b,".jld2"),res=boo)
        end
    end

    
    Theta_boots = vcat(Lambda_1_boots, Theta_delta_boots, Phi_boots, Lambda_2_boots, Psi_delta_boots, beta_boots, mu_boots, nu_boots)
    n_theta = size(Theta_boots,1)
    Theta_boots_mean = sum(Theta_boots,dims=2)/B
    cov = zeros(n_theta,n_theta)
    for b in 1:B
        cov = cov + (Theta_boots[:,b] - Theta_boots_mean)*(Theta_boots[:,b] - Theta_boots_mean)'/(B-1)
    end

    se = sqrt.(diag(cov))

    return se
end



function boots_CFAmEFA_faking(Y,X,n,p,q,restarts,K,B,std_predictors,L_str,mu,str=missing)
    std_preds = vcat(1,std_predictors); numnan = 0
    ncov = size(X,2); Lambda_1_boots = Array{Float64}(undef,p,B); Theta_delta_boots = Array{Float64}(undef,p,B); Phi_boots = Array{Float64}(undef,q*q,B); Lambda_2_boots = Array{Float64}(undef,p*K,B); Psi_delta_boots = Array{Float64}(undef,p,B); beta_boots = Array{Float64}(undef,ncov,B); mu_boots = Array{Float64}(undef,q,B); nu_boots = Array{Float64}(undef,K,B); Ez1_boots = Array{Float64}(undef,n,B);
    for b in 1:B
        println(string("boot: ",b))
        boots0 = sample(1:220,220,replace=true)
        boots1 = sample(221:735,515,replace=true)

        Y = Y[:,vcat(boots0,boots1)]
        X = X[vcat(boots0,boots1),:]
        
        
        res =  CFAmixEFA(Y, q, K, restarts, X, 2000, mu, L_str)
      

        Lambda_1_boots[:,b] = sum(reshape(res[1:(p*q)],p,q),dims=2)
        Lambda_2_boots[:,b] = res[(p*q+1):((p*q)+(p*K))]
        Theta_delta_boots[:,b] = res[((p*q)+(p*K)+1):((p*q)+(p*K)+p)]
        Psi_delta_boots[:,b] = res[((p*q)+(p*K)+p+1):((p*q)+(p*K)+p+p)]
        Phi_boots[:,b] = res[((p*q)+(p*K)+p+p+1):((p*q)+(p*K)+p+p+q^2)]
        Ez1_boots[:,b] = res[((p*q)+(p*K)+p+p+q^2+n+1+1):((p*q)+(p*K)+p+p+q^2+n+1+n)]
        beta_boots[:,b] = res[((p*q)+(p*K)+p+p+q^2+n+1+n+1):((p*q)+(p*K)+p+p+q^2+n+1+n+ncov)]./std_preds
        mu_boots[:,b] = res[((p*q)+(p*K)+p+p+q^2+n+1+n+ncov+1):((p*q)+(p*K)+p+p+q^2+n+1+n+ncov+q)]
        nu_boots[:,b] = res[((p*q)+(p*K)+p+p+q^2+n+1+n+ncov+q+1):((p*q)+(p*K)+p+p+q^2+n+1+n+ncov+q+K)]

        boo = vcat(Lambda_1_boots[:,b], Theta_delta_boots[:,b], Phi_boots[:,b], Lambda_2_boots[:,b], Psi_delta_boots[:,b], beta_boots[:,b], mu_boots[:,b], nu_boots[:,b])
        if str !== missing
            jldsave(string(str,"/boots_",b,".jld2"),res=boo)
        end
        numnan = numnan + Int(isnan(boo[1,1]))
    end

    
    
    
    Theta_boots = vcat(Lambda_1_boots, Theta_delta_boots, Phi_boots, Lambda_2_boots, Psi_delta_boots, beta_boots, mu_boots, nu_boots)
    Theta_boots = Theta_boots[:,isnan.(Theta_boots[1,:])]
    if size(Theta_boots,2) == 0
        println("All NaN, possibly increase the number of restarts")
        return Theta_boots
    end
    n_theta = size(Theta_boots,1)
    Theta_boots_mean = sum(Theta_boots,dims=2)/B
    cov = zeros(n_theta,n_theta)
    for b in 1:B
        cov = cov + (Theta_boots[:,b] - Theta_boots_mean)*(Theta_boots[:,b] - Theta_boots_mean)'/(B-1)
    end

    se = sqrt.(diag(cov))

    return vcat(se,numnan)
end



end
