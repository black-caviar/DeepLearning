import System.Random
import Data.List

import Graphics.Rendering.Chart.Easy
import Graphics.Rendering.Chart.Backend.Cairo

main = do
    seed  <- newStdGen
    print seed
    print "that was the seed"
    let rs = randomlist 10 seed
    print rs
    let k = cleanRSin (1.0,5.0) 10 seed
    print k
    ek <- newStdGen
    print (cleanRSin (1.0,5.0) 10 (mkStdGen 42))
    
    

randomlist :: Int -> StdGen -> [Int]
randomlist n = take n . unfoldr (Just . random)

sinX :: [Double] -> [Double]
sinX x = map sin x

--randN :: Int -> StdGen -> [Double]
--randN n gen = splitAt n randomRs (

cleanRSin :: (Double,Double) -> Int -> StdGen -> [Double]   
cleanRSin (a,b) n gen = do
  let x = take n (randomRs (a,b) gen)
    in sinX x
