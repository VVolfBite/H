package net.lab1024.sa.admin.module.system.earthquake.manager;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import net.lab1024.sa.admin.module.system.earthquake.dao.EarthquakeDao;
import net.lab1024.sa.admin.module.system.earthquake.domain.entity.EarthquakeEntity;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class EarthquakeManager extends ServiceImpl<EarthquakeDao, EarthquakeEntity> {
    @Autowired
    private EarthquakeDao earthquakeDao;


    /**
     * 保存震情信息
     *
     * @param earthquake
     */
    @Transactional(rollbackFor = Throwable.class)
    public void saveEarthquake(EarthquakeEntity earthquake) {
        // 保存震情信息
        earthquakeDao.insert(earthquake);
    }

    /**
     * 更新震情信息
     *
     * @param earthquake
     */
    @Transactional(rollbackFor = Throwable.class)
    public void updateEarthquake(EarthquakeEntity earthquake) {
        // 保存震情信息
        earthquakeDao.updateById(earthquake);
    }

}
