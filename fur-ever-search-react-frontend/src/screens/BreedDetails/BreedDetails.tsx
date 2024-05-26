import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';


import './BreedDetails.css';
import { MainNavigation } from '../../components/MainNavigation';
import BreedInfo from './BreedInfo';
import { FindNearestBreed } from '../../components/FindNearestBreed';
import { RecommendationPreferences } from '../../components/RecommendationPreferences'
import { breedDetailsMock } from '../../data/breedDetails';

export interface ILandings {
  landings: any
}

function BreedDetails(props: any) {

  const { breedName } = useParams<any>();

  const [breedInfo, setBreedInfo] = useState<any[]>();

  useEffect(() => {
    // Fetch data from the API endpoint
    axios.get(`http://127.0.0.1:5000/breeds/${breedName}`)
      .then(response => {
        // console.log('response', response.data)
        // Extract breed name, description, and breed image URL
        // Parse the JSON string into a JavaScript object
        const responseData = response?.data
        console.log('responseData - jashp', responseData)
        // const formattedBreeds = response.data.breeds.map((breed: any) => ({
        //   name: breed['\u00ef\u00bb\u00bfbreed'],
        //   description: breed.description,
        //   imageUrls: JSON.parse(breed.images.replace(/'/g, "\""))
        // }));
        // Set the formatted breed data to the state
        const breedInfo = responseData?.breed_info?.[0]
        console.log('breedInfo - hehe', breedInfo)
        setBreedInfo(breedInfo);
      })
      .catch(error => {
        console.error('Error fetching data:', error);
        setBreedInfo(breedDetailsMock as any);
      });
  }, []);

  console.log('breed info before return', breedInfo)

  if (!breedInfo) {
    return null
  }

  return (
    <div className='App'>
      <MainNavigation cartItemNumber={0} />
      <h1>Breed Details</h1>
      <BreedInfo breedInfo={breedInfo} />
    </div>
  );
}

export default BreedDetails;
