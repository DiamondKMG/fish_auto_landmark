library( ANTsRNet )
library( ANTsR )
library( patchMatchR )
library( tensorflow )
library( keras )
library( tfdatasets )
library( reticulate )
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
    temp[[k]] = iMath(temp[[k]], "PadImage",32) %>% ANTsRNet::padImageByFactor(8)
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
ulabs = c( 0,  unique( labelIDs$JoinWith ) )
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
ccef = categorical_focal_loss_fixed
unet = createUnetModel2D( list( NULL, NULL, 3 ),
  mode='classification', numberOfOutputs = length( ulabs ), numberOfLayers=4 )
# training system is:
# 1. train with cce
# 2. train with ccef
# 3. train with weighted cce  (optional)
# 4. train with dice (optional)
unetfn = "fish_seg.h5"
# if ( file.exists( unetfn ) )
#  unet = load_model_hdf5( unetfn, compile=FALSE  )
unet %>% compile(  loss = categorical_focal_loss_fixed,
  optimizer = optimizer_adam( lr = 1e-4  )  )
if ( ! exists( "visualize" ) ) visualize = FALSE
testLoss = c( )
ctTest = 1
for ( i in 1:50000 ) {
  k = sample( which( isTrain ), 1 )
  img = antsImageRead( imageFNS[k] )
  seg = antsImageRead( labelFNS[k], dimension = 2 ) %>% remapSegmentation()
  gg = generateData( img, seg, subSampling = sample( c(8,4),1 ) )
  unet %>% fit( gg[[1]], gg[[2]], verbose=1, epochs=2 )
  if ( ( i %% 10 == 0 ) | i == 1 ) {
    testOverlaps = rep( NA, length( which( !isTrain ) ) )
    ct = 1
    for ( kk in which( !isTrain ) ) {
      img = antsImageRead( imageFNS[kk] )
      seg = antsImageRead( labelFNS[kk], dimension = 2 )  %>% remapSegmentation()
      gg = generateData( img, seg, subSampling = 4, batch_size = 2, mySdAff = 0.01  )
      pp = predict( unet, gg[[1]] )
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
        seglow = seglow + postmapGT * (ii+1)
        if ( visualize ) {
          plot( refimg, postmap  )
          plot( refimg, postmapGT )
          }
        }
      if ( visualize ) {
        plot( refimg, usegimg )
        plot( refimg, seglow )
        }
      lom = labelOverlapMeasures( seglow, usegimg )
      testOverlaps[ ct ] = mean( lom$MeanOverlap )
      ct = ct + 1
      }
    }
    if ( mean( testOverlaps ) < min( testLoss ) ) {
      save_model_hdf5( unet, unetfn )
      }
    testLoss[ctTest] = mean( testOverlaps )
    print( testLoss[testLoss] )
    ctTest = ctTest + 1
  }
}
