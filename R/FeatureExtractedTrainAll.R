
nonpath ="F:/Research/Moonlight/FeatureExtraction/Data/All/moonlight.fasta"
moonpath = "F:/Research/Moonlight/FeatureExtraction/Data/All/nonMP.fasta"

non = ftrCOOL::AAutoCor(nonpath)
moon =ftrCOOL::AAutoCor(moonpath)
class1 = matrix (0,nrow(non),1)
non =cbind( non, class1)
class2 = matrix (1,nrow(moon),1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,file="F:/Research/Moonlight/FeatureExtraction/Data/All/AAutoCor.csv",row.names = TRUE)

#********************************************************************************************************************

non = ftrCOOL::AAKpartComposition(nonpath)
moon =ftrCOOL::AAKpartComposition(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/AAKpartComposition.csv",row.names = TRUE)

#********************************************************************************************************************

non = ftrCOOL::CkSAApair(nonpath)
moon =ftrCOOL::CkSAApair(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/CkSAApair.csv",row.names = TRUE)

#********************************************************************************************************************

non = ftrCOOL::CkSGAApair(nonpath)
moon =ftrCOOL::CkSGAApair(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/CkSGAApair.csv",row.names = TRUE)

#********************************************************************************************************************

non = ftrCOOL::CTD(nonpath)
moon =ftrCOOL::CTD(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/CTD.csv",row.names = TRUE)


#********************************************************************************************************************

non = ftrCOOL::CTDC(nonpath)
moon =ftrCOOL::CTDC(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/CTDC.csv",row.names = TRUE)

#********************************************************************************************************************

non = ftrCOOL::CTDD(nonpath)
moon =ftrCOOL::CTDD(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/CTDD.csv",row.names = TRUE)

#********************************************************************************************************************

non = ftrCOOL::CTDT(nonpath)
moon =ftrCOOL::CTDT(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/CTDC.csv",row.names = TRUE)

#********************************************************************************************************************

non = ftrCOOL::DDE(nonpath)
moon =ftrCOOL::DDE(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/DDE.csv",row.names = TRUE)
#********************************************************************************************************************


non = ftrCOOL::ExpectedValueAA(nonpath)
moon =ftrCOOL::ExpectedValueAA(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/ExpectedValueAA.csv",row.names = TRUE)

#********************************************************************************************************************

non = ftrCOOL::ExpectedValueGAA(nonpath)
moon =ftrCOOL::ExpectedValueGAA(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/ExpectedValueGAA.csv",row.names = TRUE)

#********************************************************************************************************************

non = ftrCOOL::ExpectedValueGKmerAA(nonpath)
moon =ftrCOOL::ExpectedValueGKmerAA(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/ExpectedValueGKmerAA.csv",row.names = TRUE)

#********************************************************************************************************************

non = ftrCOOL::ExpectedValueKmerAA(nonpath)
moon =ftrCOOL::ExpectedValueKmerAA(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/ExpectedValueKmerAA.csv",row.names = TRUE)

#********************************************************************************************************************

non = ftrCOOL::kAAComposition(nonpath)
moon =ftrCOOL::kAAComposition(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/kAAComposition.csv",row.names = TRUE)

#********************************************************************************************************************

non = ftrCOOL::kGAAComposition(nonpath)
moon =ftrCOOL::kGAAComposition(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/kGAAComposition.csv",row.names = TRUE)

#********************************************************************************************************************

non = ftrCOOL::GrpDDE(nonpath)
moon =ftrCOOL::GrpDDE(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/GrpDDE.csv",row.names = TRUE)


#********************************************************************************************************************

non = ftrCOOL::SAAC(nonpath)
moon =ftrCOOL::SAAC(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/SAAC.csv",row.names = TRUE)

#********************************************************************************************************************

non = ftrCOOL::SGAAC(nonpath)
moon =ftrCOOL::SGAAC(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/SGAAC.csv",row.names = TRUE)

#********************************************************************************************************************

non = ftrCOOL::QSOrder(nonpath)
moon =ftrCOOL::QSOrder(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/QSOrder.csv",row.names = TRUE)

#********************************************************************************************************************

non = ftrCOOL::SOCNumber(nonpath)
moon =ftrCOOL::SOCNumber(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/SOCNumber.csv",row.names = TRUE)

#********************************************************************************************************************

non = ftrCOOL::PseKRAAC_T1(nonpath)
moon =ftrCOOL::PseKRAAC_T1(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/PseKRAAC_T1.csv",row.names = TRUE)

#********************************************************************************************************************

non = ftrCOOL::PseKRAAC_T2(nonpath)
moon =ftrCOOL::PseKRAAC_T2(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/PseKRAAC_T2.csv",row.names = TRUE)

#********************************************************************************************************************

non = ftrCOOL::PseKRAAC_T3A(nonpath)
moon =ftrCOOL::PseKRAAC_T3A(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/PseKRAAC_T3A.csv",row.names = TRUE)

#********************************************************************************************************************

non = ftrCOOL::PseKRAAC_T3B(nonpath)
moon =ftrCOOL::PseKRAAC_T3B(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/PseKRAAC_T3B.csv",row.names = TRUE)

#********************************************************************************************************************

non = ftrCOOL::PseKRAAC_T4(nonpath)
moon =ftrCOOL::PseKRAAC_T4(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/PseKRAAC_T4.csv",row.names = TRUE)

#********************************************************************************************************************

non = ftrCOOL::PseKRAAC_T5(nonpath)
moon =ftrCOOL::PseKRAAC_T5(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/PseKRAAC_T5.csv",row.names = TRUE)

#********************************************************************************************************************

non = ftrCOOL::PseKRAAC_T6A(nonpath)
moon =ftrCOOL::PseKRAAC_T6A(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/PseKRAAC_T6A.csv",row.names = TRUE)

#********************************************************************************************************************

non = ftrCOOL::PseKRAAC_T6B(nonpath)
moon =ftrCOOL::PseKRAAC_T6B(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/PseKRAAC_T6B.csv",row.names = TRUE)


#********************************************************************************************************************

non = ftrCOOL::PseKRAAC_T7(nonpath)
moon =ftrCOOL::PseKRAAC_T7(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/PseKRAAC_T7.csv",row.names = TRUE)
#********************************************************************************************************************

non = ftrCOOL::PseKRAAC_T8(nonpath)
moon =ftrCOOL::PseKRAAC_T8(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/PseKRAAC_T8.csv",row.names = TRUE)
#********************************************************************************************************************

non = ftrCOOL::PseKRAAC_T9(nonpath)
moon =ftrCOOL::PseKRAAC_T9(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/PseKRAAC_T9.csv",row.names = TRUE)
#********************************************************************************************************************

non = ftrCOOL::PseKRAAC_T10(nonpath)
moon =ftrCOOL::PseKRAAC_T10(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/PseKRAAC_T10.csv",row.names = TRUE)
#********************************************************************************************************************

non = ftrCOOL::PseKRAAC_T11(nonpath)
moon =ftrCOOL::PseKRAAC_T11(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/PseKRAAC_T11.csv",row.names = TRUE)
#********************************************************************************************************************

non = ftrCOOL::PseKRAAC_T12(nonpath)
moon =ftrCOOL::PseKRAAC_T12(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/PseKRAAC_T12.csv",row.names = TRUE)
#********************************************************************************************************************

non = ftrCOOL::PseKRAAC_T13(nonpath)
moon =ftrCOOL::PseKRAAC_T13(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/PseKRAAC_T13.csv",row.names = TRUE)
#********************************************************************************************************************

non = ftrCOOL::PseKRAAC_T14(nonpath)
moon =ftrCOOL::PseKRAAC_T14(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/PseKRAAC_T14.csv",row.names = TRUE)
#********************************************************************************************************************

non = ftrCOOL::PseKRAAC_T15(nonpath)
moon =ftrCOOL::PseKRAAC_T15(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/PseKRAAC_T15.csv",row.names = TRUE)
#********************************************************************************************************************

non = ftrCOOL::PseKRAAC_T16(nonpath)
moon =ftrCOOL::PseKRAAC_T16(moonpath)
non =cbind( non, class1)
moon = cbind(moon,class2)
proteins = rbind(moon,non)
proteins = round(proteins,6)
write.csv(proteins,"F:/Research/Moonlight/FeatureExtraction/Data/All/PseKRAAC_T16.csv",row.names = TRUE)
