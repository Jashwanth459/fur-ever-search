import React, { useState } from 'react';
import './BreedInfo.css';
import { BreedRecommendations } from '../../components/BreedRecommendations'
import { YouTubeComponent } from '../../components/YoutubeComponent';
import { FindNearestBreed } from '../../components/FindNearestBreed';
import { sendUserInteraction } from '../../helpers/userInteraction';

const BreedInfo = ({ breedInfo }: any) => {
    console.log('breedInfo inside', breedInfo)
    const images = JSON.parse(breedInfo.images.replace(/'/g, "\""))
    const breedName = breedInfo['\u00ef\u00bb\u00bfbreed']

    const [isLiked, setIsLiked] = useState(false);
    const [isDisliked, setIsDisliked] = useState(false);

    const storedData = sessionStorage.getItem('mySessionData') || '';
    const retrievedSessionData = storedData && JSON.parse(storedData);
    const uid = retrievedSessionData?.userInfo?.user_id;
    
    console.log('uid - jashp', uid)

    const handleLikeClick = () => {
        setIsLiked(!isLiked);
        setIsDisliked(false);
        if (!isLiked) {
            if (uid) {
                sendUserInteraction(uid, breedName, 'Liked')
            }
        }
    };

    const handleDislikeClick = () => {
        setIsDisliked(!isDisliked);
        setIsLiked(false);
    };

    console.log('breedInfo - jashp', breedInfo)

    const renderTemperament = breedInfo?.temperament?.split(',').map((temperament: string, index: number) => (
        <span key={index} className="temperament">
            {temperament}
        </span>
    ));

    return (
        <div>
            <div className="breed-info-container">
                <div className="breed-info-header">
                    <img src={images[0]} alt={`${breedName} Terrier`} className="breed-image" />
                    <h1 className="breed-name">{breedName}</h1>
                </div>
                <div className="breed-info-content">
                    <p className="description">{breedInfo?.description}</p>
                    <div className="attributes">
                        <div className="attribute">
                            <strong>Temperament:</strong> {renderTemperament}
                        </div>
                        <div className="attribute">
                            <strong>Group:</strong> {breedInfo?.group}
                        </div>
                        <div className="attribute">
                            <strong>Height:</strong> {breedInfo?.min_height}" - {breedInfo?.max_height}"
                        </div>
                        <div className="attribute">
                            <strong>Weight:</strong> {breedInfo?.min_weight} lbs - {breedInfo?.max_weight} lbs
                        </div>
                        <div className="attribute">
                            <strong>Life Expectancy:</strong> {breedInfo?.min_expectancy} - {breedInfo?.max_expectancy} years
                        </div>
                    </div>
                </div>
            </div>
            <br />
            <YouTubeComponent searchKeywords={`Fascinating facts about the ${breedName}`} breedName={breedName} />
            {uid && <div className="feedback-button-container">
                <button
                    className={isLiked ? 'like-button-selected' : 'button-default'}
                    onClick={handleLikeClick}
                    style={{ marginRight: '10px' }}
                >
                    Like
                </button>
                {/* <button
                    className={isDisliked ? 'dislike-button-selected' : 'button-default'}
                    onClick={handleDislikeClick}
                >
                    Dislike
                </button> */}
                {/* <p className="message">
                    {isLiked && !isDisliked && 'Liked!'}
                    {isDisliked && !isLiked && 'Disliked!'}
                    {!isLiked && !isDisliked && 'No opinion'}
                </p> */}
            </div>}
            {/* Additional details based on the breedName */}
            <div className="horizontal-line"></div>
            <FindNearestBreed breedName={breedName} />
            <div className="horizontal-line"></div>
            <BreedRecommendations title={`Similar to ${breedName}`} breedName={breedName} />
        </div>
    );
};

export default BreedInfo;
