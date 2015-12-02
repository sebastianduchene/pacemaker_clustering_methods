library(cluster)
library(doParallel)
library(foreach)


folder_name <- commandArgs(trailingOnly = TRUE)[1]

data_sets <- dir(folder_name)
f_data<- paste0(folder_name, '/', data_sets)

#opt_ks <- vector()
#for(i in 1:10){# length(f_data)){
#    print(paste('working on simulation', i))
#    data_temp <- read.table(f_data[i], sep = ',', head = T)
#    clust_temp <- clusGap(data_temp, pam, K.max = nrow(data_temp) - 1, B  = 100)
#    opt_ks[i] <- maxSE(clust_temp$Tab[, 3], clust_temp$Tab[, 4])
#}


get_pam <- function(i){
    data_temp <- read.table(f_data[i], sep = ',', head = T)
    clust_temp <- clusGap(data_temp, pam, K.max = nrow(data_temp) - 1, B  = 100)
    return(maxSE(clust_temp$Tab[, 3], clust_temp$Tab[, 4]))
}



cl <- makeCluster(4)
registerDoParallel(cl)
opt_ks <- foreach(x = 1:10, .packages = 'cluster') %dopar% get_pam(x)
opt_ks <- as.matrix(opt_ks)
write.table(opt_ks, file = paste0(folder_name, '_pam.csv'), sep = ',')
