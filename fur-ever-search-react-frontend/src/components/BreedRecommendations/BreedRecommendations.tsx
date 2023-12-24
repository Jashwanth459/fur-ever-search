import axios from 'axios';
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import Youtube from 'react-youtube';
import { recommendationSubGroups } from '../RecommendationPreferences/helper'

import './BreedRecommendations.css'
import { RecommendationPreferences } from '../RecommendationPreferences';

const baseImgUrl = 'https://image.tmdb.org/t/p/original';

export function BreedRecommendations({ title, breedName }: any) {
    const [recommendedBreeds, setRecommendedBreeds] = useState<any>([]);
    const [recommendationPreferences, setRecommendationPreferences] = useState<any>([]);
    console.log('recommendation prefs', recommendationPreferences)

    useEffect(() => {
        // Fetch data from the API endpoint
        axios.post(`http://127.0.0.1:5000/recommendations/${breedName}`, {
            recommendationPreferences
        })
            .then(response => {
                const responseData = response?.data
                console.log('recommendedBreeds - jashp', responseData)

                // Iterate through recommendations and fetch breed info for each recommendation
                const fetchRecommendedBreedsInfo = async () => {
                    const breedInfoPromises = responseData?.recommendations.map((recommendation: string) =>
                        axios.get(`http://127.0.0.1:5000/breeds/${recommendation}`)
                    );

                    try {
                        const breedInfoResponses = await Promise.all(breedInfoPromises);
                        const recommendedBreedsData = breedInfoResponses.map(response => {
                            const responseBreedData = response.data;
                            const breedInfo = responseBreedData?.breed_info[0]
                            const images = JSON.parse(breedInfo.images.replace(/'/g, "\""))
                            const breedName = breedInfo['\u00ef\u00bb\u00bfbreed']
                            return {
                                images,
                                breedName
                            }
                        });
                        console.log('recommendedBreedsData', recommendedBreedsData)
                        setRecommendedBreeds(recommendedBreedsData)
                    } catch (error) {
                        console.error('Error fetching recommended breed info:', error);
                    }
                };

                fetchRecommendedBreedsInfo();

            })
            .catch(error => {
                console.error('Error fetching data:', error);
            });
    }, [breedName, recommendationPreferences]);

    return (
        <div>
            <RecommendationPreferences breedName={breedName} setRecommendationPreferences={setRecommendationPreferences} />
            {/* <h1 style={{marginTop: '-60px'}}>{title}</h1> */}

            <div className='recommendations_row'>
                {recommendedBreeds.map((breed: any, index: number) => (
                    <a href={`/breed-details/${breed?.breedName}`} key={index}>
                        <img alt={breed?.breedName} title={breed?.breedName} src={breed?.images[0]} />
                        <p>{breed?.breedName}</p>
                    </a>
                ))}
            </div>
            {/* {trailerUrl && <div className='youtube_embed'>
                <span className="youtube_close_button" onClick={() => { setTrailerUrl('') }}>&times;</span>
                <Youtube videoId={trailerUrl} />
            </div>} */}
        </div>
    );
}