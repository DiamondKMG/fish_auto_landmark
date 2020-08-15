library( ANTsRNet )
library( ANTsR )
library( patchMatchR )
library( tensorflow )
# message("DISABLING eager execution- WARNING!")
# tf$compat$v1$disable_eager_execution()
library( keras )
library( tfdatasets )
library( reticulate )
K <- keras::backend()
set.seed( 0 )
#####################
np <- import("numpy")
#####################
labelIDs = read.csv( "src/label_identities.csv")
nChannels = 3
remapSegmentation <- function( x ) {
  # simplify segmentations into common labels that can reliably be segmented
  newseg = x * 0
  for ( k in 1:nrow( labelIDs ) ) {
    y = labelIDs$LabelNumber[k]
    z = labelIDs$JoinWith[k]
    newseg = newseg + thresholdImage( x, y, y ) * z
  }
  newseg
}

categorical_focal_loss_fixed <- function(y_true, y_pred) {
    gamma = tf$constant( 2.00 )
    alpha = tf$constant( 0.25 )
    oneR  = tf$constant( 1.00 )
    K <- keras::backend()
    y_pred <- y_pred/K$sum(y_pred, axis = -1L, keepdims = TRUE)
    y_pred <- K$clip(y_pred, K$epsilon(), oneR - K$epsilon())
    cross_entropy = y_true * K$log(y_pred)
    gain <- alpha * K$pow( oneR - y_pred, gamma) * cross_entropy
    return(-K$sum(gain, axis = -1L))
    }

read.fcsv<-function( x, toLPS = TRUE, skip=3 ) {
  df = read.table( x, skip=skip, sep=',' )[,c(1:3,12)]
  colnames( df ) = c("id","x","y",'Label')
  return( df )
  }


#
polarX <- function(X) {
        x_svd <- svd(X)
        P <- x_svd$u %*% diag(x_svd$d) %*% t(x_svd$u)
        Z <- x_svd$u %*% t(x_svd$v)
        if (det(Z) < 0)
            Z = Z * (-1)
        return(list(P = P, Z = Z, Xtilde = P %*% Z))
    }
randAff <- function( loctx,  txtype = "Rigid", sdAffine,
      idparams, fixParams, seed ) {
      set.seed( seed )
      idim = 2
      noisemat = stats::rnorm(length(idparams), mean = 0, sd = sdAffine)
      if (txtype == "Translation")
        noisemat[1:(length(idparams) - idim )] = 0
      idparams = idparams + noisemat
      idmat = matrix(idparams[1:(length(idparams) - idim )],
                  ncol = idim )
      idmat = polarX(idmat)
              if (txtype == "Rigid")
                  idmat = idmat$Z
              if (txtype == "Affine")
                  idmat = idmat$Xtilde
              if (txtype == "ScaleShear")
                  idmat = idmat$P
              idparams[1:(length(idparams) - idim )] = as.numeric(idmat)
      setAntsrTransformParameters(loctx, idparams)
      setAntsrTransformFixedParameters( loctx, fixParams )
      return(loctx)
      }

#
randomRotateImage <- function( image, sdAff=0.1, seed = 0 ) {
  fixedParams = getCenterOfMass( image * 0 + 1 )
  loctx <- createAntsrTransform(precision = "float",
    type = "AffineTransform", dimension = image@dimension  )
  setAntsrTransformFixedParameters(loctx, fixedParams)
  idparams = getAntsrTransformParameters( loctx )
  setAntsrTransformParameters( loctx, idparams )
  setAntsrTransformFixedParameters(loctx, fixedParams)
  loctx = randAff( loctx, sdAffine=sdAff, txtype = 'Affine',
    idparams = idparams, fixParams = fixedParams, seed = seed )
  imageR = applyAntsrTransformToImage( loctx, image, image,
      interpolation = "nearestNeighbor" )
  return( imageR )
}


generateData <- function( imgIn, segger,
  batch_size = 16, mySdAff=0.15,
  subSampling = 8 ) {
  imgSub = resampleImage( imgIn, dim( imgIn )/subSampling, useVoxels=T)
  temp = splitChannels( imgSub )
  doReflect = rnorm( 1 ) < 0
  reflectMat = '/tmp/reflect.mat'
  for ( k in 1:nChannels ) {
    temp[[k]] = iMath(temp[[k]], "PadImage", 64 ) %>% ANTsRNet::padImageByFactor( 8 )
    if ( k == 1 ) myreflection = reflectionMatrix( temp[[k]], 0, reflectMat )
    if ( doReflect ) temp[[k]] = antsApplyTransforms( temp[[k]], temp[[k]], reflectMat )
    }
  imgSub = mergeChannels( temp )
  segSub = resampleImageToTarget( segger, splitChannels(imgSub)[[1]], interpType='nearestNeighbor' )
  if ( doReflect ) segSub = antsApplyTransforms( segSub, segSub, reflectMat, interpolator='nearestNeighbor' )
  X = array( dim = c( batch_size, dim( imgSub  ), nChannels ) )
  kMeansK = length( ulabs )
  Xm = array( dim = c( batch_size, dim( imgSub  ), kMeansK ) )
  for ( k in 1:batch_size ) {
    seeder = Sys.time()
    splitter = splitChannels( imgSub )
    for ( j in 1:nChannels ) {
      rr = randomRotateImage( splitter[[j]], sdAff=mySdAff, seed = seeder ) %>% iMath( "Normalize" )
      if ( imgSub@dimension == 2 ) X[k,,,j] = as.array( rr  )
      if ( imgSub@dimension == 3 ) X[k,,,,j] = as.array( rr  )
    }
    rrm = randomRotateImage( segSub, sdAff=mySdAff, seed = seeder )
    for ( pp in 1:kMeansK ) {
      temp = thresholdImage( rrm, ulabs[pp], ulabs[pp] )
      if ( imgSub@dimension == 2 ) Xm[k,,,pp]    = as.array( temp  )
      if ( imgSub@dimension == 3 ) Xm[k,,,,pp]   = as.array( temp  )
      }
    }
  list( X,  Xm )
}

toLuminance <- function( x ) {
  temp = splitChannels( x )
  w = c(0.299,0.587,0.114)
  temp =  temp[[1]]*w[1]+temp[[2]]*w[2]+temp[[3]]*w[3]
  msk = getMask( temp, 5, 253 )
  temp2 = -( temp * msk )
  iMath( temp2, "Normalize" ) * 255 * msk
  }

######## here, we find the unique labels #########
ulabs = sort( unique( c( 0,  labelIDs$JoinWith ) ) )
if ( ! exists( "ulabs" ) ) {
  fns=Sys.glob("labelmaps/*")
  ulabs=sort(unique(antsImageRead( fns[1] )))
  for ( k in 1:length(fns) ) ulabs=intersect( ulabs, unique(antsImageRead( fns[1] )) )
}
imageFNS = Sys.glob( "images/*jpg" )
labelFNS = Sys.glob( "labelmaps/*nrrd" )
pointFNS = Sys.glob( "fcsv/*fcsv" )
# show example
img = antsImageRead( imageFNS[1] )
imgl = toLuminance( img )
imgl = resampleImage( imgl, dim( imgl )/8, useVoxels=T )
pts = read.fcsv( pointFNS[1] )
ptsi = makePointsImage( pts[,c("x","y")], imgl, radius=25 )
# plot( imgl, ptsi )
seg = antsImageRead( labelFNS[1], dimension = 2 )
seg2 = remapSegmentation( seg )
# plot( imgl, resampleImageToTarget( seg2, imgl, interpType='nearestNeighbor' ) )
nPoints = nrow( pts )
isTrain = rep( TRUE, length(  labelFNS  ) )
isTrain[ sample( 1:length( labelFNS ), 10 )  ] = FALSE

# plot( as.antsImage( gg[[1]][1,,,1] ), as.antsImage( gg[[2]][1,,,3] ) )

################################################################################
######################## set up the custom network #############################
################################################################################
cce = tf$keras$losses$CategoricalCrossentropy()
mse = tf$keras$losses$MeanSquaredError()
mae = tf$keras$losses$MeanAbsoluteError()
dicer = multilabel_dice_coefficient
ccef = categorical_focal_loss_fixed
mywts = c( 0.2, 0.2, 0.25, 0.25,
  0.5, 0.5, 0.3, 0.35, 0.22 )
weightsTensor <- K$variable( mywts/sum(mywts) )
weighted_categorical_crossentropy_fixed <- function( y_true, y_pred )
   {
   y_pred <- y_pred / K$sum( y_pred, axis = -1L, keepdims = TRUE )
   y_pred <- K$clip( y_pred, K$epsilon(), 1.0 - K$epsilon() )
   loss <- y_true * K$log( y_pred ) * weightsTensor
   loss <- -K$sum( loss, axis = -1L )
   return( loss )
   }
ccew = weighted_categorical_crossentropy_fixed
unet = createUnetModel2D( list( NULL, NULL, 3 ),
  mode='classification', numberOfOutputs = length( ulabs ), numberOfLayers=4 )
# training system is:
# 1. train with cce
# 2. train with ccef
# 3. train with weighted cce  (optional)
# 4. train with dice (optional)
unetfn = "models/fish_seg.h5"
csvfn = gsub(  "h5", "csv" , unetfn )
if ( file.exists( unetfn ) ) {
  unet = load_model_hdf5( unetfn, compile=FALSE  )
  trainingDataFrame = read.csv( csvfn )
} else {
  trainingDataFrame = data.frame(
    TrainLoss = NA,
    TestDice  = NA
  )
}
if ( ! exists( "visualize" ) ) visualize = FALSE
i = nrow( trainingDataFrame )
message("should implement exponential averaging to prevent subject-specific over-fitting")
message("should consider defining sample weights in order to counter potential imbalance")
message("in the representation of different appearance/types (related to species/genus stuff)")
message("but need more biological knowledge to decide this.")
message("should consider whether weighted CCE is needed.")
unet %>% compile(  loss = ccef,
  optimizer = optimizer_adam( lr = 1e-4  )  )
if ( i > 4000 ) {
  unet %>% compile(  loss = ccew,
    optimizer = optimizer_adam( lr = 1e-4  )  )
  }
# if ( i > 4000 ) {
#  unet %>% compile(  loss = dicer,
#      optimizer = optimizer_adam( lr = 1e-4  )  )
#  }
# FIXME - need to estimate weights empirically
#    unet %>% compile(  loss = weighted cce,
#      optimizer = optimizer_adam( lr = 1e-4  )  )
for ( i in i:50000 ) {
  trainingDataFrame[i,]=NA
  if ( i < 100 ) {
    mySubSam = 12
    nEpch = 10
  }
  if ( i >= 100 ) {
    mySubSam = sample( c( 12, 8 ), 1 )
    nEpch = 4
  }
  if ( i >= 200 ) {
    mySubSam = 8
    nEpch = 2
    }
  if ( i >= 10000 ) {
    mySubSam = sample( c( 12, 8, 6 ), 1 )
    nEpch = 1
    }
  k = sample( which( isTrain ), 1 )
  img = antsImageRead( imageFNS[k] )
  seg = antsImageRead( labelFNS[k], dimension = 2 ) %>% remapSegmentation()
  print( paste("Train On:", k,  imageFNS[k] ) )
  gg = generateData( img, seg, batch_size = 16, subSampling = mySubSam, mySdAff=0.15 )
  tracking <- unet %>% fit( gg[[1]], gg[[2]], verbose = 0, epochs=nEpch )
  trainingDataFrame[i,"TrainLoss"] = head( tracking[2]$metrics$loss , 1 )
  if ( i > 10  & (i %% 10 == 0 ) ) {
    plot( ts( trainingDataFrame[,"TrainLoss"] ) )
    print( paste(i, "RecentLossMean:", mean( tail( trainingDataFrame[,"TrainLoss"] ) ) ) )
  }
  checker = 10
  if ( ( i %% checker == 0 ) | i == 1 ) {
    testOverlaps = rep( NA, length( which( !isTrain ) ) )
    ct = 1
    for ( kk in which( !isTrain ) ) {
      mySubSam = 8
      img = antsImageRead( imageFNS[kk] )
      seg = antsImageRead( labelFNS[kk], dimension = 2 )  %>% remapSegmentation()
      gg = generateData( img, seg, subSampling = mySubSam, batch_size = 1, mySdAff = 0.0  )
      pp = predict( unet, tf$cast( gg[[1]], "float32" ) )
      refimg = as.antsImage( gg[[1]][1,,,1] )
      dd = decodeUnet( pp, refimg )
      segvec = imageListToMatrix( dd[[1]], refimg * 0 + 1 )
      segvec = apply( segvec, FUN=which.max, MARGIN=2 )
      usegimg = makeImage( refimg * 0 + 1, segvec )
      layout( matrix( 1:24, nrow = 4, byrow=T ) )
      seglow = refimg * 0
      for ( ii in 1:length(ulabs) ) {
        postmap = as.antsImage( pp[1,,,ii] ) %>% antsCopyImageInfo2( seglow )
        postmapGT = as.antsImage( gg[[2]][1,,,ii] ) %>% antsCopyImageInfo2( seglow )
        seglow = seglow + postmapGT * ii
        if ( visualize & max( postmap ) > 0.1 ) {
#          plot( refimg, postmap  )
#          plot( refimg, postmapGT )
          }
        }
      if ( visualize ) {
        layout( matrix( 1:2, nrow = 1, byrow=T ) )
        plot( refimg, usegimg, window.overlay=c(2,max(seglow)), alpha=0.5 )
        plot( refimg, seglow, window.overlay=c(2,max(seglow)), alpha=0.5 )
        }
      lom = labelOverlapMeasures( seglow, usegimg )
      testOverlaps[ ct ] = mean( lom$MeanOverlap[-1] )
      ct = ct + 1
      }
    ctTest = sum( !is.na( trainingDataFrame[,"TestDice"] ) )
    if ( ctTest > 5 ) {
      if ( mean( testOverlaps, na.rm=T ) >= max( trainingDataFrame[,"TestDice"], na.rm=T ) ) {
        save_model_hdf5( unet, unetfn )
        }
      }
    trainingDataFrame[i,"TestDice"] = mean( testOverlaps, na.rm=T )
    print( paste( "Test", ctTest, ":", trainingDataFrame[i,"TestDice"]  ) )
    if ( ctTest > 10 ) plot( ts( na.omit( trainingDataFrame[,"TestDice"] ) ) )
    }
  write.csv( trainingDataFrame, csvfn, row.names = FALSE )
  }
