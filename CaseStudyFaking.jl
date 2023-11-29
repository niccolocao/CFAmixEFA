##############################################################################
#################### Case study 1: Faking behaviuor ##########################
##############################################################################

push!(LOAD_PATH,string(pwd(),"/MixCFAEFA/"))
using MixCFAEFA, CSV, StatsBase, DataFrames, DelimitedFiles, Distributions, Random, JLD2, Clustering, LinearAlgebra, Base.Threads, StatFiles


#for data import (original dataset retrived from https://osf.io/e3scf) and data cleaning see R script casestudies.R, from which we obtained faking_data.csv 
Y_mix = CSV.read(string("/home/nc/MEGA/CFA+EFA/application/faking/faking_data.csv"), DataFrame,header=1)[:,2:end];

#decouple fakers from honest
Y_nonfakers = Y_mix[:,1:44]; Y_nonfakers[!,:z1] .= 1;

#reverse items in honest
rev = [4,31,6,13,14,15,17,22,27,35,34,33,8,36,38,39,40] .+2;
for i in 1:17
    Y_nonfakers[:,rev[i]] = 6 .- Y_nonfakers[:,rev[i]]
end

#prepare submatrix of fakers by selecting their covariates values, bind them to the response vectors, and assigning 0 as their identificative number (the "known" latent assignment)
Y_fakers = Y_mix[:,1:2]; Y_fakers = hcat(Y_fakers,Y_mix[:,45:end]); Y_fakers[!,:z1] .= 0;
#same columns names also for the honest matrix
Y_fakers = rename!(Y_fakers,names(Y_nonfakers));

#reverse coding of items in faking condition
rev = [4,31,6,13,14,15,17,22,27,35,34,33,8,36,38,39,40] .+2;
for i in 1:7
    Y_fakers[:,rev[i]] = 6 .- Y_fakers[:,rev[i]]
end

#sub-sampling (different statistical units) to obtain a matrix with 30\% of fakers
Random.seed!(183972274); #make it reproducible 
select_1 = sample(1:735,515,replace=false);
Y_nonfakers = Y_nonfakers[select_1,:];
Y_fakers = Y_fakers[Not(select_1),:];

#bind fakers and honest
Y = vcat(Y_fakers, Y_nonfakers);

#reordering of the items by common latent variables (i.e., items measuring the same latent variable are together)
Y = Y[:,vcat(45,1,2,3,5,7,9,18,22,26,40,4,6,8,19,24,25,27,34,36,10,14,16,30,31,35,39,41,42,43,11,12,15,17,21,37,38,44,13,20,23,28,29,32,33)];

#vector of true (latent) assignments
z = Y[:,1];
writedlm(string(pwd(),"/z_faking.csv"),z)

#matrix for logistric regression on the mixture parameter
gender = Y[:,2] .-1;
age = Y[:,3];
X = Matrix{Float64}(hcat(ones(size(gender)),age)); ncov = size(X,2);
std_pred = std(X[:,2]); #standard deviation of age
X[:,2] = (X[:,2].-mean(X[:,2]))./std(X[:,2]);

Y = Matrix{Float64}(Y[:,4:end]);
#some minimal variation and standardization
Random.seed!(2739797);
for i in 1:size(Y,1)
    Y[i,:] = Y[i,:] + rand(Uniform(-0.02,0.02),size(Y,2))
end
for i in 1:size(Y,2)
    Y[:,i]=(Y[:,i].-mean(Y,dims=1)[i])/std(Y,dims=1)[i]
end

#final matrix with 42 x 732 (items x observations)
Y0 = Y';
Y = Matrix{Float64}(Y0);
writedlm(string(pwd(),"~/final_faking_data.csv"),Y);


p = size(Y,1);
n = size(Y,2);
q = 5;
K = 1;


#definition of the CFA latent structure
L_str = hcat(cat(repeat([1],8), repeat([0],9), repeat([0],10), repeat([0],8), repeat([0],7),dims=1),cat(repeat([0],8), repeat([1],9), repeat([0],10), repeat([0],8), repeat([0],7),dims=1),cat(repeat([0],8), repeat([0],9), repeat([1],10), repeat([0],8), repeat([0],7),dims=1),cat(repeat([0],8), repeat([0],9), repeat([0],10), repeat([1],8), repeat([0],7),dims=1),cat(repeat([0],8), repeat([0],9), repeat([0],10), repeat([0],8), repeat([1],7),dims=1));

#CFA+EFA model estiamtion
res = CFAmixEFA(Y,q,K,500,X,2000,0,L_str)

#bootstrap standard errors note that the evaluation would take (very) long time
bse = boots_CFAmEFA_faking(Y,X,n,p,q,K,1500,std_pred,L_str,0)  #std_pred is the vector of standard deviation of the covariates, 0 indicates that the CFA means are setted to zero


#save predicted latent memberships
Ez = round.(Int64,res[((p*q)+(p*K)+p+p+q^2+n+1+1):((p*q)+(p*K)+p+p+q^2+n+1+n)])
writedlm(string(pwd(),"/Ez_faking.csv"),Ez);


# Classification performances:
classperf(z,Ez)



# Template for the model selection procedure: change path for store the estimated models
mods = 6
#models estimation and storing
for K in 1:mods
    res = CFAmixEFA(Y, q, K, 500, ones(n,1), 2000, 0, L_str)
    jldsave(string("/home/nc/MEGA/CFA+EFA/application/Arias/model_selection/res_null_",K,".jld2"),res = res)
    res = CFAmixEFA(Y, q, K, 500, X, 2000, 0, L_str)
    jldsave(string("/home/nc/MEGA/CFA+EFA/application/Arias/model_selection/res_age_",K,".jld2"),res = res)
end

#matrices initialization
CC = Array{Float64}(undef,mods); BACC = Array{Float64}(undef,mods); MCC = Array{Float64}(undef,mods); SE = Array{Float64}(undef,mods); SP = Array{Float64}(undef,mods);  ARI = Array{Float64}(undef,mods); Precision = Array{Float64}(undef,mods); Recall = Array{Float64}(undef,mods); BIC = Array{Float64}(undef,mods); ssBIC = Array{Float64}(undef,mods);  CLC = Array{Float64}(undef,mods); ICLBIC = Array{Float64}(undef,mods); CAIC = Array{Float64}(undef,mods); LLIK = Array{Float64}(undef,mods); AIC = Array{Float64}(undef,mods); AICc = Array{Float64}(undef,mods);  perplexity = Array{Float64}(undef,mods);  Entropy = Array{Float64}(undef,mods); 
measures = zeros(6,12,2);l=0;p = size(Y,1)
# for loop to compute the model selection indices and classification metrics
for j in ["null" "age"]
    l=l+1
    if j=="null"
        ncov = 1
    elseif j=="age"
        ncov = 2
    end
    for i in 1:mods
        println(i)
        K = i
        res = JLD2.load(string("/home/nc/MEGA/CFA+EFA/application/Arias/model_selection/res_",j,"_",i,".jld2"))["res"]


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


        # Calculation classification metrics
        z = round.(Int64,z)
        CC[i] = sum(Ez1.== z)/n 
        TrueNegatives = sum(Ez1.== 1 .&& z.==  1)
        FalseNegatives = sum(Ez1.!= 1 .&& z .== 1)
        FalsePositives = sum(Ez1.== 1 .&& z .!= 1)
        TruePositives = sum(Ez1.!= 1 .&& z .!= 1)
        
        SP[i] = TrueNegatives/(TrueNegatives+FalsePositives)
        SE[i] = TruePositives/(TruePositives+FalseNegatives)
        MCC[i] = (TruePositives*TrueNegatives-FalsePositives*FalseNegatives)/sqrt((TruePositives+FalsePositives)*(TruePositives+FalseNegatives)*(TrueNegatives+FalsePositives)*(TrueNegatives+FalseNegatives))
        BACC[i] = (TruePositives/(TruePositives+FalseNegatives)+(TrueNegatives/(TrueNegatives+FalsePositives))) / 2
        
        # Calculation selection criteria
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
        P1p[findall(P1p.>1)].=0.9999999999999
        P0 = zeros(n)
        P1p[findall(P1p.>1)].=0.9999999999999
        for j=1:n
            P0[j] = (1-P1p[j])*log(1-P1p[j])
            P1[j] = P1p[j]*log(P1p[j])
        end
        P0[isnan.(P0)].=0.0
        Ek = - (sum(P1) + sum(P0))
        Entropy[i] = 1-(Ek/(n*log(2)))
        
        npar = p+p*K+p+p+(q*(q-1))/2+ncov+K    


        LLIK[i] = Expllik
        AIC[i] = 2*npar .-2*Expllik
        CAIC[i] = npar *(log(n)+1).-2*Expllik
        BIC[i] = -2*Expllik.+npar *log(n)
        ssBIC[i] = -2*Expllik.+log((n+2)/24)*npar  
        CLC[i] = -2*Expllik+2*Ek 
        ICLBIC[i] = -2*Expllik+log(n)*npar +2*Ek
    end
    measures[1:6,1:12,l] = hcat(LLIK,AIC,CAIC,BIC,ssBIC, CLC,ICLBIC, Entropy,BACC,MCC,SE,SP)
end
res_selection = vcat(["LLIK" "AIC" "CAIC" "BIC" "ssBIC" " CLC" "ICLBIC" "Entropy" "BACC" "MCC" "SE" "SP"],vcat(measures[:,:,1],measures[:,:,2]))


