import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { sendUserInteraction } from '../../helpers/userInteraction';
import Youtube from 'react-youtube';

export const YouTubeComponent = ({ searchKeywords, breedName }: any) => {
    const [videos, setVideos] = useState([]);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await axios.get('https://www.googleapis.com/youtube/v3/search', {
                    params: {
                        key: 'AIzaSyDCSjv9E93UuLxdFkq5bfMKD7npM2Qgqw8', // Not required for public searches, but leave it empty if you don't have an API key
                        q: searchKeywords, // The search keywords or query
                        part: 'snippet',
                        type: 'video',
                        maxResults: 1, // Number of videos to retrieve
                    },
                });

                console.log('youtube response', response.data)
                setVideos(response.data.items);
            } catch (error) {
                console.error('Error fetching YouTube data:', error);
            }
        };

        fetchData();
    }, [searchKeywords]);

    const youtubeHandler = (e: any) => {
        console.log('Clicked on the YouTube video');
        console.log('event e', e)
        const storedData = sessionStorage.getItem('mySessionData') || '';
        const retrievedSessionData = storedData && JSON.parse(storedData);
        const uid = retrievedSessionData?.userInfo?.user_id;
        if(uid) {
            sendUserInteraction(uid, breedName, 'Video');
        }
    }
    console.log('yputube compoent hit')

    return (
        <div className="youtube-container">
            {videos.map((video: any) => (
                <div key={video.id.videoId} className="video-item">
                    <h2>{video.snippet.title}</h2>
                    {/* <div onClick={youtubeHandler}>
                        <iframe
                            width="560"
                            height="315"
                            src={`https://www.youtube.com/embed/${video.id.videoId}`}
                            title={video.snippet.title}
                            allowFullScreen
                            onClick={youtubeHandler}
                        />
                    </div> */}
                    <Youtube
                        videoId={video.id.videoId}
                        title={video.snippet.title}
                        onPlay={youtubeHandler}
                    />
                </div>
            ))}
        </div>
    );
};

export default YouTubeComponent;
