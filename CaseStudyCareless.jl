##############################################################################
#################### Case study 2: Careless responding #######################
##############################################################################

push!(LOAD_PATH,string(pwd(),"/MixCFAEFA/"))
using MixCFAEFA, CSV, StatsBase, DataFrames, DelimitedFiles, Distributions, Random, JLD2, Clustering, LinearAlgebra, Base.Threads, StatFiles


#downloaded dataset from https://osf.io/e3scf
data = DataFrame(load("~/DATA_SAMPLE_1.sav"))[:,2:39];
data = dropmissing(data, disallowmissing=true);
n = size(data,1);


#model matrix for logistic regression with standardized predictors: intercept in column 1, gender in column 2, age in column 3
X = hcat(ones(size(data,1)),Matrix{Float64}(data[:,1:2]))
std_pred1 = std(X[:,3]); 
X[:,3] = (X[:,3].-mean(X[:,3]))./std(X[:,3]);

#standardized data matrix of items x subjects (60x425)
Y = Matrix{Float64}(data[:,3:end]);

Random.seed!(2739797);
#some minimal variation
for i in 1:size(Y,1)
    Y[i,:] = Y[i,:] + rand(Uniform(-0.02,0.02),size(Y,2))
end
#standarization of observed variables
for i in 1:size(Y,2)
    Y[:,i]=(Y[:,i].-mean(Y,dims=1)[i])/std(Y,dims=1)[i]
end

#standardized data matrix of items x subjects (36x708)
Y = Matrix(Y');

q = 3; #number of CFA factors
K = 17; #number of EFA factors

#definition of the CFA latent structure
L_str = hcat(cat(repeat([1],12),repeat([0],12), repeat([0],12),dims=1),
             cat(repeat([0],12),repeat([1],12), repeat([0],12),dims=1),
             cat(repeat([0],12),repeat([0],12), repeat([1],12),dims=1));

#CFA+EFA estimation
res = CFAmixEFA(Y, q, K, 500, X, 2000, 1, L_str)

#bootstrap standard errors, note that the evaluation would take long time
bse = boots_CFAmEFA_careless(Y,X,n,p,q,K,1500,std_pred,L_str1,1)  #std_pred is the vector of standard deviation of the covariates, 1 indicates the estimation of CFA means

#extract the predicted latent values and save them 
Ez = round.(Int64,res[((p*q)+(p*K)+p+p+q^2+n+1+1):((p*q)+(p*K)+p+p+q^2+n+1+n)])
writedlm(string(pwd(),"/Ez.csv"),Ez)



# Template for the model selection procedure: change path for store the estimated models
mods = 20
#models estimation and storing
for K in 1:mods
    res = CFAmixEFA(Y, q, K, 500, ones(n,1), 2000, 1, L_str)
    jldsave(string("~/model_selection/res_null_",K,".jld2"),res = res)
    res = CFAmixEFA(Y, q, K, 500, X[:,1:2], 2000, 1, L_str)
    jldsave(string("~/model_selection/res_gender_",K,".jld2"),res = res)
    res = CFAmixEFA(Y, q, K, 500, X[:,vcat(1,3)], 2000, 1, L_str)
    jldsave(string("~/model_selection/res_age_",K,".jld2"),res = res)
    res = CFAmixEFA(Y, q, K, 500, X, 2000, 1, L_str)
    jldsave(string("~/model_selection/res_complete_",K,".jld2"),res = res)
end

#matrix initialization 
BIC = Array{Float64}(undef,mods); ssBIC = Array{Float64}(undef,mods);  CLC = Array{Float64}(undef,mods); ICLBIC = Array{Float64}(undef,mods); CAIC = Array{Float64}(undef,mods); LLIK = Array{Float64}(undef,mods); AIC = Array{Float64}(undef,mods); AICc = Array{Float64}(undef,mods);  perplexity = Array{Float64}(undef,mods);  Entropy = Array{Float64}(undef,mods); 
measures = zeros(20,8,4);l=0;p = size(Y,1)
# for loop to compute the model selection indices
for j in ["null" "gender" "age" "complete"]
    l=l+1
    if j=="null"
        ncov = 1
    elseif j=="gender" || j=="age"
        ncov = 2
    elseif j=="complete"
        ncov = 3
    end
    for i in 1:mods
        println(i)
        K = i
        res = JLD2.load(string("~/model_selection/res_",j,"_",i,".jld2"))["res"]


        Lambda_1 = reshape(res[1:(p*q)],p,q)
        Lambda_2 = reshape(res[(p*q+1):((p*q)+(p*K))],p,K)
        Theta_delta = Diagonal(res[((p*q)+(p*K)+1):((p*q)+(p*K)+p)])
        Psi_delta = Diagonal(res[((p*q)+(p*K)+p+1):((p*q)+(p*K)+p+p)])
        Phi = reshape(res[((p*q)+(p*K)+p+p+1):((p*q)+(p*K)+p+p+q^2)],q,q)
        kappa = res[((p*q)+(p*K)+p+p+q^2+1):((p*q)+(p*K)+p+p+q^2+n)]
        Expllik = reduce(vcat,res[((p*q)+(p*K)+p+p+q^2+n+1):((p*q)+(p*K)+p+p+q^2+n+1)])
        Ez1 = res[((p*q)+(p*K)+p+p+q^2+n+1+1):((p*q)+(p*K)+p+p+q^2+n+1+n)]
        if Ez1[1] !== NaN
            Ez1 = round.(Int64,Ez1)
        end
        betas = res[((p*q)+(p*K)+p+p+q^2+n+1+n+1):((p*q)+(p*K)+p+p+q^2+n+1+n+ncov)]
        mu = res[((p*q)+(p*K)+p+p+q^2+n+1+n+ncov+1):((p*q)+(p*K)+p+p+q^2+n+1+n+ncov+q)]
        nu = res[((p*q)+(p*K)+p+p+q^2+n+1+n+ncov+q+1):((p*q)+(p*K)+p+p+q^2+n+1+n+ncov+q+K)]

        inv_lemma_1 = inv_lemma_1_fun(q,Lambda_1,Theta_delta,Phi)
        inv_lemma_2 = inv_lemma_2_fun(K,Lambda_2,Psi_delta)      
        det_lemma_1 = det_lemma_1_fun(Lambda_1,Theta_delta,Phi)
        det_lemma_2 = det_lemma_2_fun(K,Lambda_2,Psi_delta)
        b1 = Phi*Lambda_1'*inv(Lambda_1*Phi*Lambda_1'+Theta_delta);
        b2 = Lambda_2'*inv(Lambda_2*Lambda_2' + Psi_delta);
        P1p = zeros(n)
        P1 = zeros(n)
        for i in 1:n
            P1p[i] = ((kappa[i] .* ((exp((-1/2) * Y[:,i]'*inv_lemma_1*Y[:,i]))/(sqrt((2 .* pi)^(p) *det_lemma_1))))/((kappa[i] .* ((exp((-1/2) * Y[:,i]'*inv_lemma_1*Y[:,i]))/(sqrt((2 .* pi)^(p) *det_lemma_1)))).+((exp((-1/2) * Y[:,i]'*inv_lemma_2*Y[:,i]))/(sqrt((2 .* pi)^(p) .*det_lemma_2))).-(kappa[i] .*((exp((-1/2) * Y[:,i]'*inv_lemma_2*Y[:,i]))/(sqrt((2 .* pi)^(p) .*det_lemma_2))))))
        end
        P1p = P1p .+ 1e-7
        P0 = zeros(n)
        P1p[findall(P1p.>1)].=0.9999999999999
        for j=1:n
            P0[j] = (1-P1p[j])*log(1-P1p[j])
            P1[j] = P1p[j]*log(P1p[j])
        end
        P0[isnan.(P0)].=0.0
        Ek = - (sum(P1) + sum(P0))
        Entropy[i] = 1-(Ek/(n*log(2)))
            

        npar = p+p*K+p+p+(q*(q-1))/2+ncov+K+q


        LLIK[i] = Expllik
        AIC[i] = 2*npar .-2*Expllik
        CAIC[i] = npar *(log(n)+1).-2*Expllik
        BIC[i] = -2*Expllik.+npar *log(n)
        ssBIC[i] = -2*Expllik.+log((n+2)/24)*npar  
        CLC[i] = -2*Expllik+2*Ek 
        ICLBIC[i] = -2*Expllik+log(n)*npar +2*Ek
    end
    measures[1:20,1:8,l] = hcat(LLIK,AIC,CAIC,BIC,ssBIC, CLC,ICLBIC, Entropy)
end
res_selection = vcat(["LLIK" "AIC" "CAIC" "BIC" "ssBIC" " CLC" "ICLBIC" "Entropy"],vcat(measures[:,:,1],measures[:,:,2],measures[:,:,3],measures[:,:,4]))
    


