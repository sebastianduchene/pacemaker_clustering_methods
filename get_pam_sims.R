library(cluster)

folder_name <- commandArgs(trailingOnly = TRUE)[1]

data_sets <- dir(folder_name)
f_data<- paste0(folder_name, '/', data_sets)

opt_ks <- vector()
for(i in 1:length(f_data)){
    print(paste('working on simulation', i))
    data_temp <- read.table(f_data[i], sep = ',', head = T)
    clust_temp <- clusGap(data_temp, pam, K.max = nrow(data_temp) - 1, B  = 100)
    opt_ks[i] <- maxSE(clust_temp$Tab[, 3], clust_temp$Tab[, 4])
}

write.table(opt_ks, file = paste0(folder_name, '.csv'), sep = ',')
