####################################################################################
######################## Case study 1: Faking behaviour ############################
####################################################################################


#############################################################################################################
# downloaded data from https://osf.io/e3scf
# note that the original file is .txt
# first steps to make the file readable in Julia
da = read.csv("~/2_data_ApplicantsFakingBigFive.csv", sep=";")

# data cleaning  
faking_data = da[c(2:45,130:171)]
faking_data = na.omit(faking_data)

# usable file .csv
write.csv(faking_data,"~/faking_data.csv")
#############################################################################################################


# import the final faking data with 30% of faking and the true (latent) memberships to faking and honest subsamples (contained in z vector)
dat = read.csv("~/final_faking_data.csv",sep="\t",header = F)
z = read.csv("~/z.csv",sep="\t",header = F)

#naming items
fak = cor(dat[z==0,]); colnames(fak) = paste("Item",1:42);rownames(fak) = paste("Item",1:42)
hon = cor(dat[z==1,]); colnames(hon) = paste("Item",1:42);rownames(hon) = paste("Item",1:42)

#hierarchical clustering based on Ward.D2 distance
clus_fak = as.dendrogram(hclust(dist(fak), method = "ward.D2"))
clus_hon = as.dendrogram(hclust(dist(hon), method = "ward.D2"))

library(dendextend)
dl <- dendlist(
  clus_hon %>% 
    set("branches_lty", 1)  %>% set("labels_cex", 1) %>% set("branches_lwd", 1.3),
  clus_fak %>% 
    set("branches_lty", 1) %>% set("labels_cex", 1) %>% set("branches_lwd", 1.3)
)

#tanglegran if the dendrograms
tanglegram(dl, highlight_distinct_edges  = F, highlight_branches_lwd=FALSE,lwd=3, margin_inner = 3.85, margin_outer = 0.15, main_left ="Honest", main_right = "Faking",cex.main=2.8,lab.cex=1.3)

#entanglement index
entanglement(clus_fak,clus_hon)

library(RColorBrewer)
cor_mat_cfa = cor(dat[z==1,])
cor_mat_efa = cor(dat[z==0,])
cor_mat_tot = cor(dat)
colnames(cor_mat_cfa) = paste("Item",1:42);colnames(cor_mat_efa) = paste("Item",1:42);colnames(cor_mat_tot) = paste("Item",1:42)
rownames(cor_mat_cfa) = paste("Item",1:42);rownames(cor_mat_efa) = paste("Item",1:42);rownames(cor_mat_tot) = paste("Item",1:42)

#Heatmaps:
itt = c("1",rep("   ",3),"5",rep("   ",4),"10",rep("   ",4),"15",rep("   ",4),"20",rep("   ",4),"25",rep("   ",4),"30",rep("   ",4),"35",rep("   ",4),"40",rep("   ",2))
pheatmap::pheatmap(cor_mat_cfa,cluster_rows = F,cluster_cols = F,fontsize = 20,colorRampPalette(rev(brewer.pal(n=11, name="RdBu")))(100),breaks=seq(-1, 1, length.out=101),cellwidth = 10.1,cellheight = 10.1,width = 4,height = 4,show_rownames = T, 
                   labels_row =itt,labels_col = itt,legend = F)
pheatmap::pheatmap(cor_mat_efa,cluster_rows = F,cluster_cols = F,fontsize = 20,colorRampPalette(rev(brewer.pal(n=11, name="RdBu")))(100),breaks=seq(-1, 1, length.out=101),cellwidth = 10.1,cellheight = 10.1,width = 4,height = 4,show_rownames = T, 
                   labels_row =itt,labels_col = itt,legend = F)
pheatmap::pheatmap(cor_mat_tot,cluster_rows = F,cluster_cols = F,fontsize = 20,colorRampPalette(rev(brewer.pal(n=11, name="RdBu")))(100),breaks=seq(-1, 1, length.out=101),cellwidth = 10.09,cellheight = 10.09,width = 4,height = 4,show_rownames = T, 
                   labels_row =itt,labels_col = itt)



# CFA on the observed dataset
colnames(hon)=c(paste0("A",seq(1,8)),paste0("B",seq(1,9)),paste0("C",seq(1,10)),paste0("D",seq(1,8)),paste0("E",seq(1,7)))
colnames(fak)=c(paste0("A",seq(1,8)),paste0("B",seq(1,9)),paste0("C",seq(1,10)),paste0("D",seq(1,8)),paste0("E",seq(1,7)))
colnames(dat)=c(paste0("A",seq(1,8)),paste0("B",seq(1,9)),paste0("C",seq(1,10)),paste0("D",seq(1,8)),paste0("E",seq(1,7)))


model = paste(
  paste("A",paste(paste0("A",1:8),collapse = "+"),sep = " =~ "),
  paste("B",paste(paste0("B",1:9),collapse = "+"),sep = " =~ "),
  paste("C",paste(paste0("C",1:10),collapse = "+"),sep = " =~ "),
  paste("D",paste(paste0("D",1:8),collapse = "+"),sep = " =~ "),
  paste("E",paste(paste0("E",1:7),collapse = "+"),sep = " =~ "),
  sep = " \n ")


library(lavaan)
library(psych)
cfa_mix = cfa(model = model,data = dat,estimator="ML")

fitmeasures(object = cfa_mix,fit.measures = c("AIC","RMSEA","CFI","NFI","chisq","df","npar"))


