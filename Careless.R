

####################################################################################
######################## Case study 2: Careless responding #########################
####################################################################################
library(RColorBrewer)
library(lavaan)
library(psych)
library(haven)
data <- read_sav("~/DATA_SAMPLE_1.sav")
data = na.omit(data)
data$Sex = data$Sex-1
data = data[,1:39]

# CFA on the observed data
dat = data[,4:39]
colnames(dat)=c(paste0("E",seq(1,12)),paste0("C",seq(1,12)),paste0("S",seq(1,12)))

model = paste(
  paste("E",paste(paste0("E",1:12),collapse = "+"),sep = " =~ "),
  paste("C",paste(paste0("C",1:12),collapse = "+"),sep = " =~ "),
  paste("S",paste(paste0("S",1:12),collapse = "+"),sep = " =~ "),
  sep = " \n ")

cfa_mix = cfa(model = model,data = dat,estimator="ML")

fitmeasures(object = fit_mix,fit.measures = c("AIC","RMSEA","CFI","NFI","chisq","df","npar"))



# vector of predicted latent values of the estimated model
z = read.csv("~/Ez.csv",header = F)[,1]

# labels
itt = c("1",rep("   ",4),"6",rep("   ",5),"12",rep("   ",5),"18",rep("   ",5),"24",rep("   ",5),"28",rep("   ",5),"36")

# heatmap CFA component (unbiased subjects)
cor_mat_cfa = cor(data[z==1,4:39])
pheatmap::pheatmap(cor_mat_cfa,cluster_rows = F,cluster_cols = F,fontsize = 20,colorRampPalette(rev(brewer.pal(n=11, name="RdBu")))(100),breaks=seq(-1, 1, length.out=101),cellwidth = 12,cellheight = 12,width = 4,height = 4,show_rownames = T, 
                   labels_row =itt,labels_col = itt,legend = F)

# heatmap EFA component (biased subjects)
cor_mat_efa = cor(data[z==0,4:39])
pheatmap::pheatmap(cor_mat_efa,cluster_rows = F,cluster_cols = F,fontsize = 20,colorRampPalette(rev(brewer.pal(n=11, name="RdBu")))(100),breaks=seq(-1, 1, length.out=101),cellwidth = 12,cellheight = 12,width = 4,height = 4,show_rownames = T, 
                   labels_row =itt,labels_col = itt)

