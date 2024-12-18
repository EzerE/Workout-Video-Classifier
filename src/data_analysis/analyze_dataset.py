def get_video_duration(file_path):
    try:
        video = cv2.VideoCapture(file_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        video.release()
        return round(duration, 2)
    except Exception as e:
        print(f"Error getting duration for {file_path}: {e}")
        return 0

def get_file_info(base_folder):
    folder_stats = defaultdict(list)
    
    for root, _, files in os.walk(base_folder):
        exercise_name = os.path.basename(root)
        
        for file in files:
            if file.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.wmv')):
                file_path = os.path.join(root, file)
                try:
                    size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 2)
                    mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    duration = get_video_duration(file_path)
                    
                    folder_stats[exercise_name].append({
                        'name': file,
                        'size_mb': size_mb,
                        'format': file.split('.')[-1],
                        'modified': mod_time.strftime('%Y-%m-%d %H:%M'),
                        'duration': duration
                    })
                except Exception as e:
                    print(f"Error processing {file}: {e}")
    
    return folder_stats

def format_duration(seconds):
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"{minutes}:{remaining_seconds:02d}"

def create_summary_dataframes(folder_stats):
    summary_data = []
    detailed_data = []
    
    for exercise, videos in folder_stats.items():
        if videos:
            formats = {}
            total_size = 0
            total_duration = 0
            sizes = []
            durations = []
            
            for video in videos:
                format_type = video['format']
                formats[format_type] = formats.get(format_type, 0) + 1
                total_size += video['size_mb']
                total_duration += video['duration']
                sizes.append(video['size_mb'])
                durations.append(video['duration'])
            
            summary_data.append({
                'Exercise': exercise,
                'Total Videos': len(videos),
                'Formats': ', '.join(f"{fmt}: {count}" for fmt, count in formats.items()),
                'Total Size (MB)': round(total_size, 2),
                'Size Std (MB)': round(pd.Series(sizes).std(), 2) if len(sizes) > 1 else 0,
                'Total Duration': format_duration(total_duration),
                'Duration Std': format_duration(pd.Series(durations).std()) if len(durations) > 1 else "0:00"
            })
            
            for video in videos:
                detailed_data.append({
                    'Exercise': exercise,
                    'Video Name': video['name'],
                    'Format': video['format'],
                    'Size (MB)': video['size_mb'],
                    'Duration': format_duration(video['duration']),
                    'Last Modified': video['modified']
                })
    
    summary_df = pd.DataFrame(summary_data)
    detailed_df = pd.DataFrame(detailed_data)
    
    # Add total row to summary DataFrame
    total_row = {
        'Exercise': 'Total',
        'Total Videos': summary_df['Total Videos'].sum(),
        'Formats': 'All',
        'Total Size (MB)': summary_df['Total Size (MB)'].sum(),
        'Size Std (MB)': summary_df['Size Std (MB)'].mean(),  # Average of standard deviations
        'Total Duration': format_duration(sum(pd.Series([int(x.split(':')[0])*60 + int(x.split(':')[1]) 
                                               for x in summary_df['Total Duration']]))),
        'Duration Std': format_duration(pd.Series([int(x.split(':')[0])*60 + int(x.split(':')[1]) 
                                                 for x in summary_df['Duration Std']]).mean())
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)
    
    return summary_df, detailed_df

# Usage example:
base_folder = r"C:\Users\user\Documents\Workout"
folder_stats = get_file_info(base_folder)
summary_df, detailed_df = create_summary_dataframes(folder_stats)

# Display the DataFrames (optional)
print("\nExercise Summary:")
print(summary_df.to_string())
print("\nDetailed Video List:")
print(detailed_df.to_string())


def time_to_seconds(time_str):
    minutes, seconds = map(int, time_str.split(':'))
    return minutes * 60 + seconds

detailed_df['Duration_Seconds'] = detailed_df['Duration'].apply(time_to_seconds)

# Set figure size
fig_size = (12, 6)
# 1. Bar plot of video count by exercise
plt.figure(figsize=fig_size)
count_data = detailed_df['Exercise'].value_counts()
bars = plt.bar(count_data.index, count_data.values)
plt.title('Number of Videos per Exercise', pad=15, fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Exercise')
plt.ylabel('Count')
# Add value labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom')
plt.tight_layout()
plt.savefig(r'C:\Users\user\Documents\k_fold_CNN_LSTM_landmark\figs\videos_per_exercise.png')
plt.show()

# New addition: Single boxplot for all video durations
plt.figure(figsize=fig_size)
sns.boxplot(data=detailed_df, y='Duration_Seconds')
plt.title('Overall Video Duration Distribution', pad=15, fontsize=12)
plt.ylabel('Duration (seconds)')
plt.tight_layout()
plt.savefig(r'C:\Users\user\Documents\k_fold_CNN_LSTM_landmark\figs\overall_duration_distribution.png')
plt.show()

# 2. Box plot for Duration by Exercise
plt.figure(figsize=fig_size)
sns.boxplot(data=detailed_df, x='Exercise', y='Duration_Seconds')
plt.title('Video Duration Distribution by Exercise', pad=15, fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Exercise')
plt.ylabel('Duration (seconds)')
plt.tight_layout()
plt.savefig(r'C:\Users\user\Documents\k_fold_CNN_LSTM_landmark\figs\duration_by_exercise.png')
plt.show()

plt.figure(figsize=fig_size)
sns.boxplot(data=detailed_df[detailed_df['Duration_Seconds']<60], y='Duration_Seconds', x='Exercise') 
plt.title('Video Duration Distribution (< 60 seconds)', pad=15, fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Exercise')
plt.ylabel('Duration (seconds)')
plt.tight_layout()
plt.savefig(r'C:\Users\user\Documents\k_fold_CNN_LSTM_landmark\figs\duration_by_exercise_under_60s.png')
plt.show()

# 3. Box plot for Size by Exercise
plt.figure(figsize=fig_size)
sns.boxplot(data=detailed_df, x='Exercise', y='Size (MB)')
plt.title('Video Size Distribution by Exercise', pad=15, fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Exercise')
plt.ylabel('Size (MB)')
plt.tight_layout()
plt.savefig(r'C:\Users\user\Documents\k_fold_CNN_LSTM_landmark\figs\size_by_exercise.png')
plt.show()
