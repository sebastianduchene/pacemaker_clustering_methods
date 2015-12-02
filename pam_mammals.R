library(cluster)

scale8 <- svd(m8)
plot(scale$u[, 1], scale$u[, 2])

sims_8 <- dir('sim_mammals_8')

out_sims_8 <- vector()
for(f in sims_8){
    print(f)
    m8 <- read.table(paste0('sim_mammals_8/', f), sep = ',',  head  = T, row.names = 1)
    c8 <- clusGap(m8, FUN = pam, K.max = 20, B = 10)
    out_sims_8 <- c(out_sims_8, maxSE(c8$Tab[, 3], c8$Tab[, 4]))
}

sims_2 <- dir('sim_mammals_2')
sims_2 <- sims_2[1:50)
out_sims_2 <- vector()
for(f in sims_2){
    print(f)
    m2 <- read.table(paste0('sim_mammals_2/', f), sep = ',', head = T, row.names = 1)
    c2 <- clusGap(m2, FUN = pam, K.max = 20, B = 10)
    out_sims_2 <- c(out_sims_2, maxSE(c2$Tab[, 3], c2$Tab[, 4]))
    print(out_sims_2)
}

