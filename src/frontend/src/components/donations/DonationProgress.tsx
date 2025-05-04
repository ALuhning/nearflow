interface DonationProgressProps {
    progress: number; // value between 0 and 100
  }
  
  export default function DonationProgress({ progress }: DonationProgressProps) {
    return (
      <div className="relative h-[200px] w-6 rounded bg-zinc-200 dark:bg-zinc-800 flex flex-col justify-end items-center mt-1">
        {/* Labels */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 text-sm mb-1">
          🎯
        </div>
        <div className="absolute top-1/4 left-full ml-2 text-xs">75%</div>
        <div className="absolute top-1/2 left-full ml-2 text-xs">50%</div>
        <div className="absolute top-3/4 left-full ml-2 text-xs">25%</div>
  
        {/* Fill */}
        <div
          className="bg-green-500 w-full rounded-b"
          style={{ height: `${progress}%`, transition: "height 0.5s ease" }}
        />
      </div>
    );
  }
  