aster <- 10^read.table('asteraceae_matrix.csv', head = T, sep = ',')
head(aster)
write.table(aster, file = 'asteraceae_matrix.csv', sep = ',')
